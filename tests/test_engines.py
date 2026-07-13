"""
Unit tests for the core engines.

Pure-function tests (no network) cover the maths; a few data-backed tests run the
statistical layer on the committed market_daily.csv. Run:  python3 -m pytest -q
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from gex_engine import bs_gamma
from positioning import max_pain, gamma_by_strike, apn_zones, zero_gamma_flip
from calendar_layer import catalyst_read
from flow_engine import add_flow_features, significant_threshold
from stats_spine import regime_two_sample, purged_walk_forward, garch_persistence
from anomalies import add_forward_returns, add_regime_labels


# ---------------- pure maths (no network) ----------------
def test_bs_gamma_positive_and_expires_to_zero():
    assert bs_gamma(60000, 60000, 0.1, 0.5) > 0
    assert bs_gamma(60000, 60000, 0.0, 0.5) == 0.0     # expired
    assert bs_gamma(60000, 60000, 0.1, 0.0) == 0.0     # zero vol


def test_bs_gamma_peaks_near_atm():
    atm = bs_gamma(60000, 60000, 0.1, 0.5)
    otm = bs_gamma(60000, 120000, 0.1, 0.5)
    assert atm > otm


def _rec(strike, kind, oi):
    return {"expiry": datetime(2026, 12, 25, tzinfo=timezone.utc), "strike": strike,
            "kind": kind, "oi": oi, "iv": 50.0, "T": 0.1, "gex": oi}


def test_max_pain_hand_computed():
    # call OI at 70k, put OI at 50k -> least intrinsic paid out at 50k
    recs = [_rec(70000, "C", 1000), _rec(50000, "P", 10)]
    assert max_pain(recs) == 50000


def test_zones_and_walls_signs():
    # heavy call gamma high, heavy put gamma low
    recs = [_rec(65000, "C", 500), _rec(66000, "C", 400),
            _rec(58000, "P", 500), _rec(60000, "P", 300)]
    strikes, call, put, net = gamma_by_strike(recs)
    z = apn_zones(strikes, call, put, net)
    assert z["P1"] in (65000, 66000)      # top positive-gamma strike is a call strike
    assert z["N1"] in (58000, 60000)      # most negative is a put strike
    assert call.sum() > 0 and put.sum() > 0


def test_zero_gamma_flip_returns_level_or_none():
    recs = [_rec(65000, "C", 500), _rec(58000, "P", 500)]
    zg = zero_gamma_flip(recs, spot=61000)
    assert zg is None or 45000 < zg < 80000


# ---------------- catalyst logic ----------------
def test_catalyst_no_event():
    r = catalyst_read(True, None)
    assert r["has_catalyst"] is False


def test_catalyst_pin_plus_imminent_is_break_risk():
    ev = {"title": "CPI m/m", "hours_until": 5.0}
    r = catalyst_read(True, ev)
    assert r["level"] == "break-risk" and r["imminent"] is True


def test_catalyst_far_event_is_quiet():
    ev = {"title": "NFP", "hours_until": 200.0}
    r = catalyst_read(True, ev)
    assert r["level"] == "quiet"


# ---------------- data-backed statistical layer ----------------
@pytest.fixture(scope="module")
def market():
    df = pd.read_csv("data/market_daily.csv", parse_dates=["date"])
    return add_flow_features(add_regime_labels(add_forward_returns(df)))


def test_significant_threshold_valid_probs(market):
    r = significant_threshold(market)
    assert 0.0 <= r["P_big_move_calm"] <= 1.0
    assert 0.0 <= r["P_big_move_high_flow"] <= 1.0


def test_vol_clustering_is_significant(market):
    r = regime_two_sample(market, "fwd_rv_3")
    assert r["welch_p"] < 0.05                 # the robust finding
    assert r["mean_elevated"] > r["mean_calm"]


def test_purged_cv_returns_briers(market):
    r = purged_walk_forward(market, "fwd_ret_3", ["elevated", "dvol_chg_z", "dvol"])
    assert "brier_model" in r and "brier_baserate" in r
    assert 0.0 < r["brier_model"] < 0.5


def test_garch_persistence_in_range(market):
    g = garch_persistence(market)
    assert 0.0 <= g["persistence"] <= 1.05
