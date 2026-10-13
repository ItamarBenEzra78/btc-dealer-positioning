"""
Tests for the engines and the statistical layer.

Pure-function tests (no network) cover the maths and the validation plumbing;
data-backed tests run the statistical layer on the frozen analysis sample
data/sample_2024-07_2026-07.csv (a copy of market_daily.csv at the time the
documented numbers were produced, so run_all.py refreshing market_daily.csv
never breaks them). Run:  python3 -m pytest -q
"""

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from anomalies import add_forward_returns, add_regime_labels
from calendar_layer import catalyst_read
from flow_engine import add_flow_features, cusum_events, kyle_lambda, significant_threshold
from gex_engine import bs_gamma, parse_instrument
from positioning import apn_zones, gamma_by_strike, max_pain, zero_gamma_flip
from stats_spine import (
    REGIME_FEATURES,
    event_study,
    garch_persistence,
    horizon_from_target,
    naive_walk_forward,
    purged_walk_forward,
    regime_two_sample,
    return_persistence,
    walk_forward_brier,
    walk_forward_splits,
)
from validate import within

SAMPLE = "data/sample_2024-07_2026-07.csv"


# ---------------- pure maths (no network) ----------------
def test_bs_gamma_positive_and_expires_to_zero():
    assert bs_gamma(60000, 60000, 0.1, 0.5) > 0
    assert bs_gamma(60000, 60000, 0.0, 0.5) == 0.0     # expired
    assert bs_gamma(60000, 60000, 0.1, 0.0) == 0.0     # zero vol


def test_bs_gamma_peaks_near_atm():
    atm = bs_gamma(60000, 60000, 0.1, 0.5)
    otm = bs_gamma(60000, 120000, 0.1, 0.5)
    assert atm > otm


def test_parse_instrument_round_trip():
    expiry, strike, kind = parse_instrument("BTC-28AUG26-46000-P")
    assert expiry == datetime(2026, 8, 28, 8, 0, tzinfo=timezone.utc)
    assert strike == 46000.0 and kind == "P"
    name = f"BTC-{expiry.day}{expiry.strftime('%b').upper()}{expiry.year % 100}-{int(strike)}-{kind}"
    assert name == "BTC-28AUG26-46000-P"


def _rec(strike, kind, oi, iv=50.0, T=0.1):
    return {"expiry": datetime(2026, 12, 25, tzinfo=timezone.utc), "strike": strike,
            "kind": kind, "oi": oi, "iv": iv, "T": T, "gex": oi}


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


def test_zero_gamma_flip_symmetric_chain():
    """One call at K1, one put at K2, equal OI/IV/T: the dealer-signed gamma
    crosses zero where the two Black-Scholes gammas are equal, i.e. at
    S* = sqrt(K1*K2) * exp(-sigma^2 T / 2). The grid search must find it."""
    k1, k2, sigma, T = 50000.0, 70000.0, 0.5, 0.1
    recs = [_rec(k1, "C", 100, iv=sigma * 100, T=T), _rec(k2, "P", 100, iv=sigma * 100, T=T)]
    expected = math.sqrt(k1 * k2) * math.exp(-0.5 * sigma ** 2 * T)
    zg = zero_gamma_flip(recs, spot=60000)
    assert zg is not None and k1 < zg < k2
    assert abs(zg - expected) / expected < 0.01


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


# ---------------- feature construction ----------------
def _toy_frame(closes, dvol=None):
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=len(closes)),
                       "close": closes})
    df["logret"] = np.log(df["close"]).diff()
    df["dvol"] = dvol if dvol is not None else np.nan
    return df


def test_add_forward_returns_hand_built():
    closes = [100, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171, 214.358881, 235.7947691]
    df = add_forward_returns(_toy_frame(closes), horizons=(1, 3))
    # constant 10% growth: every forward log return is h * ln(1.1)
    assert df["fwd_ret_1"].iloc[0] == pytest.approx(math.log(1.1))
    assert df["fwd_ret_3"].iloc[2] == pytest.approx(3 * math.log(1.1))
    assert df["fwd_ret_3"].iloc[6] == pytest.approx(math.log(235.7947691 / 177.1561))
    # the last h rows have no forward window
    assert df["fwd_ret_1"].iloc[-1:].isna().all()
    assert df["fwd_ret_3"].iloc[-3:].isna().all()
    assert df["fwd_ret_3"].iloc[:-3].notna().all()
    # constant returns -> zero forward realized vol; horizon 1 has no rv column
    assert df["fwd_rv_3"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert "fwd_rv_1" not in df.columns


def test_add_regime_labels_nan_dvol_gives_nan_regime():
    n = 60
    dvol = [np.nan] * 30 + list(np.linspace(40, 70, 30))
    df = add_regime_labels(_toy_frame(np.linspace(100, 120, n), dvol), dvol_window=90)
    reg = df["dvol_regime"]
    assert reg.iloc[:30].isna().all()                       # no DVOL -> no regime
    assert reg.iloc[30:49].isna().all()                     # < 20 obs for the median -> no regime
    assert reg.iloc[49:].isin(["elevated", "calm"]).all()   # labelled once the median exists
    assert (df.loc[reg.isna(), "elevated"] == 0).all()
    # a rising series sits above its trailing median once labelled
    assert (reg.iloc[49:] == "elevated").all()


def test_cusum_events_known_case():
    # +2 steps: cumulative 2,4,6 -> crosses h=5 at index 2, resets, crosses again at 5
    s = [2, 2, 2, 2, 2, 2, -3, -3, 0]
    assert cusum_events(s, h=5.0) == [2, 5, 7]
    assert cusum_events([0.0] * 10, h=5.0) == []
    assert cusum_events([1.0, np.nan, 5.0], h=5.0) == [2]   # NaN treated as 0


def test_event_study_synthetic_spike():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.001, 300)
    r[150] = 0.05                                    # one +5% day
    df = pd.DataFrame({"logret": r})
    out = event_study(df, [150], ret_col="logret", pre=3, post=5, est_window=60)
    assert out is not None
    tab, n = out
    assert n == 1
    row = tab.set_index("offset")
    assert row.loc[0, "mean_AR"] == pytest.approx(0.05, abs=0.002)
    assert abs(row.loc[-1, "mean_AR"]) < 0.005
    assert row.loc[5, "CAR"] == pytest.approx(0.05, abs=0.01)
    # an event without a full 60-day history or a full post window is skipped
    assert event_study(df, [10], pre=3, post=5, est_window=60) is None
    assert event_study(df, [298], pre=3, post=5, est_window=60) is None


def test_within_handles_missing_and_zero():
    assert within(100, 101, 0.02) is True
    assert within(100, 110, 0.02) is False
    assert within(None, 100, 0.02) is False
    assert within(100, 0, 0.02) is False


# ---------------- walk-forward validation ----------------
@pytest.mark.parametrize("n", [100, 477, 731])
@pytest.mark.parametrize("horizon", [1, 3, 7])
def test_walk_forward_splits_never_overlap(n, horizon):
    splits = list(walk_forward_splits(n, n_folds=5, horizon=horizon))
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.max() + horizon < te.min()          # no label window reaches the test block
        assert len(np.intersect1d(tr, te)) == 0
        assert tr.min() == 0                          # expanding window
    tests = np.concatenate([te for _, te in splits])
    assert len(np.unique(tests)) == len(tests)        # test blocks are disjoint


def test_naive_splits_have_no_gap():
    for tr, te in walk_forward_splits(200, n_folds=4, horizon=3, gap=0):
        assert tr.max() + 1 == te.min()


def test_horizon_parsed_from_target_and_mismatch_raises():
    assert horizon_from_target("fwd_ret_3") == 3
    assert horizon_from_target("fwd_ret_7") == 7
    with pytest.raises(ValueError):
        horizon_from_target("fwd_ret")
    df = pd.DataFrame({"fwd_ret_3": np.zeros(50), "x": np.zeros(50)})
    with pytest.raises(ValueError):
        purged_walk_forward(df, "fwd_ret_3", ["x"], horizon=1)


def test_leak_canary_gap_removes_label_overlap_skill():
    """Overlapping forward labels leak across a gap-less train/test boundary.

    Labels are signs of 20-day forward sums of iid noise, so adjacent labels
    share 19/20 of their inputs. A model that memorises by time proximity
    (5-NN on the time index) scores far below the base-rate Brier when the gap
    is removed, purely from that overlap. With a horizon-length gap the nearest
    training labels share no inputs with the test labels and the illusion of
    skill disappears.
    """
    rng = np.random.default_rng(0)
    n, h = 400, 20
    r = rng.standard_normal(n + h)
    y = np.array([int(r[t + 1:t + h + 1].sum() > 0) for t in range(n)])
    X = np.arange(n, dtype=float).reshape(-1, 1)

    def knn():
        return KNeighborsClassifier(n_neighbors=5)

    leaky = walk_forward_brier(X, y, horizon=h, gap=0, n_folds=80, model_factory=knn, min_train=10)
    purged = walk_forward_brier(X, y, horizon=h, gap=h, n_folds=80, model_factory=knn, min_train=10)
    assert leaky["brier_model"] < leaky["brier_baserate"] - 0.05      # fake skill
    assert purged["brier_model"] >= purged["brier_baserate"]          # gone with the gap


# ---------------- data-backed statistical layer (frozen sample) ----------------
@pytest.fixture(scope="module")
def market():
    df = pd.read_csv(SAMPLE, parse_dates=["date"])
    return add_flow_features(add_regime_labels(add_forward_returns(df)))


def test_sample_coverage(market):
    assert len(market) == 731
    labelled = market["dvol_regime"].notna()
    assert int(labelled.sum()) == 481
    assert market["dvol"].isna().sum() == 231
    assert str(market.loc[labelled, "date"].min().date()) == "2025-03-20"


def test_significant_threshold_valid_probs(market):
    r = significant_threshold(market)
    assert 0.0 <= r["P_big_move_calm"] <= 1.0
    assert 0.0 <= r["P_big_move_high_flow"] <= 1.0
    assert r["P_big_move_high_flow"] > r["P_big_move_calm"]


def test_vol_clustering_is_significant(market):
    r = regime_two_sample(market, "fwd_rv_3")
    assert r["welch_p"] < 0.001                 # the robust finding
    assert r["n_elevated"] + r["n_calm"] <= 481  # regime-labelled rows only
    assert r["mean_elevated"] == pytest.approx(45.3, abs=0.1)
    assert r["mean_calm"] == pytest.approx(27.5, abs=0.1)


def test_directional_bias_not_significant(market):
    r = regime_two_sample(market, "fwd_ret_3")
    assert r["welch_p"] > 0.05


def test_calm_mean_reversion_is_null(market):
    # documented null: the earlier "calm tapes mean-revert" result came from NaN -> 'calm' labelling
    rp = return_persistence(market)
    assert rp["calm"]["p"] > 0.1
    assert abs(rp["calm"]["autocorr_lag1"]) < 0.05
    assert rp["calm"]["n"] == 298 and rp["elevated"]["n"] == 156


def test_walk_forward_briers_match_docs(market):
    pw = purged_walk_forward(market, "fwd_ret_3", REGIME_FEATURES)
    nw = naive_walk_forward(market, "fwd_ret_3", REGIME_FEATURES)
    assert pw["n_oos"] == nw["n_oos"] == 395
    assert pw["brier_model"] == pytest.approx(0.1869, abs=5e-4)
    assert pw["brier_baserate"] == pytest.approx(0.1915, abs=5e-4)
    assert nw["brier_model"] == pytest.approx(0.1927, abs=5e-4)
    assert abs(pw["brier_baserate"] - pw["brier_model"]) < 0.01      # negligible edge


def test_kyle_predictive_is_null(market):
    kl = kyle_lambda(market)
    assert kl["kyle_pred_p"] > 0.05
    assert kl["volvol_p"] < 0.001
    assert kl["amihud"] > 0


def test_garch_persistence_in_range(market):
    g = garch_persistence(market)
    assert 0.0 <= g["persistence"] <= 1.05
    assert g["persistence"] == pytest.approx(0.952, abs=0.005)
