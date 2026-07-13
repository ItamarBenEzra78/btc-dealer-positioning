"""
anomalies.py — Turn the article's phenomena into detectable events.

Two tiers, split honestly by what free history allows:

  BACKTESTABLE NOW (deep history via price + DVOL)
    - vol_regime      : DVOL-based regime label (proxy for the gamma regime)
    - dvol_spike      : standardised jump in implied vol -> an event to study
    These let us test the article's core mechanical claim *today*: does market
    behaviour (forward realized vol, return persistence) differ by vol regime?

  LIVE DETECTORS (GEX-native, accrue as the snapshot DB grows)
    - pin_setup       : encodes the article's exact max-pain pin conditions
                        (positive gamma + low DTE + price near max pain)
    The true gamma-regime / pin backtest matures via data_layer's forward collector.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# BACKTESTABLE NOW
# ----------------------------------------------------------------------------
def add_forward_returns(df, horizons=(1, 3, 7)):
    df = df.copy()
    logc = np.log(df["close"])
    for h in horizons:
        df[f"fwd_ret_{h}"] = logc.shift(-h) - logc
        # forward realized vol = std of logret[t+1 .. t+h], annualised %
        # rolling(h).std() at t+h covers t+1..t+h; shift(-h) brings it back to t
        df[f"fwd_rv_{h}"] = df["logret"].rolling(h).std().shift(-h) * np.sqrt(365) * 100
    return df


def add_regime_labels(df, dvol_window=90):
    """Label each day's implied-vol regime relative to its trailing median.

    'elevated' vs 'calm' is our observable proxy for the dealer gamma regime:
    the article's mechanism predicts calmer, stickier tape when dealers dampen
    and wilder tape when they amplify.
    """
    df = df.copy()
    med = df["dvol"].rolling(dvol_window, min_periods=20).median()
    df["dvol_regime"] = np.where(df["dvol"] > med, "elevated", "calm")
    df["elevated"] = (df["dvol_regime"] == "elevated").astype(int)
    df["dvol_chg"] = df["dvol"].diff()
    mu = df["dvol_chg"].rolling(dvol_window, min_periods=20).mean()
    sd = df["dvol_chg"].rolling(dvol_window, min_periods=20).std()
    df["dvol_chg_z"] = (df["dvol_chg"] - mu) / sd
    return df


def detect_dvol_spikes(df, z_thresh=2.0):
    """Days where implied vol jumped abnormally — discrete events for an event study."""
    return df.index[df["dvol_chg_z"].abs() > z_thresh].tolist()


# ----------------------------------------------------------------------------
# LIVE DETECTORS (GEX-native)
# ----------------------------------------------------------------------------
def pin_setup(snap, near_pct=0.015):
    """Does the current snapshot meet the article's max-pain pin conditions?

    Returns each condition + an overall flag, so the dashboard can show *why*.
    Conditions (from the article): positive gamma regime, low DTE (0-3),
    and spot already near the (near-expiry) max pain.
    """
    spot = snap["spot"]
    mp = snap.get("max_pain_near")
    dte = snap.get("near_dte")
    conds = {
        "positive_gamma": snap["regime"] == "positive",
        "low_dte": dte is not None and dte <= 3,
        "near_max_pain": mp is not None and abs(spot - mp) / spot <= near_pct,
    }
    return {
        "conditions": conds,
        "conditions_met": sum(conds.values()),
        "pin_likely": all(conds.values()),
        "distance_to_max_pain_pct": round((spot - mp) / spot * 100, 2) if mp else None,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/market_daily.csv", parse_dates=["date"])
    df = add_forward_returns(df)
    df = add_regime_labels(df)
    spikes = detect_dvol_spikes(df)

    print(f"Rows: {len(df)}")
    reg = df["dvol_regime"].value_counts()
    print(f"Regime split: {reg.to_dict()}")
    print(f"DVOL spike events (|z|>2): {len(spikes)}")
    print("\nSample (recent):")
    cols = ["date", "close", "dvol", "dvol_regime", "dvol_chg_z", "fwd_ret_3"]
    print(df[cols].tail(5).to_string(index=False))

    from positioning import snapshot
    ps = pin_setup(snapshot("BTC"))
    print(f"\nLive pin check: {ps['conditions_met']}/3 conditions, "
          f"pin_likely={ps['pin_likely']}, dist_to_maxpain={ps['distance_to_max_pain_pct']}%")
    print(f"  {ps['conditions']}")
