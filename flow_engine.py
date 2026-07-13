"""
flow_engine.py — Money flow + the significant-threshold model (L2, the core).

Two parts, split by data availability (same honesty as L1):

  LIVE  options net-premium flow from Deribit taker trades
        (are takers paying up for calls or puts, buying or selling premium).

  BACKTESTABLE NOW  the "significant flow threshold" on 2y of daily data:
    - kyle_lambda()   price impact per unit of directional volume (Kyle's λ / Amihud)
    - cusum_events()  CUSUM filter: when does CUMULATIVE flow cross a significant level
                      (this is the literal "how much flow is enough to matter")
    - evaluated with an event study + a logistic P(big move | flow anomaly).

This directly implements the user's emphasised question: what size of flow is
statistically significant enough to move price?
"""

import numpy as np
import pandas as pd
from scipy import stats

from gex_engine import _get, DERIBIT


# ----------------------------------------------------------------------------
# LIVE — options net-premium flow
# ----------------------------------------------------------------------------
def live_option_flow(currency="BTC", count=1000):
    """Aggregate recent option taker trades into net premium flow (calls vs puts)."""
    r = _get(f"{DERIBIT}/get_last_trades_by_currency?currency={currency}&kind=option&count={count}")
    trades = r["trades"]
    call_prem = put_prem = 0.0
    for t in trades:
        sign = 1.0 if t["direction"] == "buy" else -1.0     # taker aggressor side
        notional = t["price"] * t["amount"]                  # premium in BTC terms
        if t["instrument_name"].endswith("-C"):
            call_prem += sign * notional
        else:
            put_prem += sign * notional
    net = call_prem + put_prem
    return {
        "n_trades": len(trades),
        "net_call_premium": round(call_prem, 3),
        "net_put_premium": round(put_prem, 3),
        "net_premium_flow": round(net, 3),
        "flow_tilt": "calls" if call_prem > abs(put_prem) else "puts" if put_prem < -abs(call_prem) else "mixed",
    }


# ----------------------------------------------------------------------------
# BACKTESTABLE — significant flow threshold
# ----------------------------------------------------------------------------
def add_flow_features(df):
    """Directional volume + standardized flow, from OHLCV we already have."""
    df = df.copy()
    # intraday direction as aggressor proxy (avoids circularity with close-close return)
    df["signed_vol"] = np.sign(df["close"] - df["open"]) * df["volume"]
    mu = df["volume"].rolling(30, min_periods=10).mean()
    sd = df["volume"].rolling(30, min_periods=10).std()
    df["vol_z"] = (df["volume"] - mu) / sd
    smu = df["signed_vol"].rolling(30, min_periods=10).mean()
    ssd = df["signed_vol"].rolling(30, min_periods=10).std()
    df["signed_vol_z"] = (df["signed_vol"] - smu) / ssd
    return df


def kyle_lambda(df):
    """Price impact of directional volume.

    Contemporaneous λ shares the close price with the target, so its R² is partly
    mechanical — reported but not trusted. The honest test is PREDICTIVE: does
    today's signed flow move TOMORROW's return? That can't be circular.
    """
    d = df.dropna(subset=["logret", "signed_vol"]).copy()
    scale = d["volume"].abs().mean()
    x = d["signed_vol"].values / scale
    y = d["logret"].values
    contemp = stats.linregress(x, y)

    # predictive: flow_t -> return_{t+1}
    d["fwd1"] = d["logret"].shift(-1)
    dp = d.dropna(subset=["fwd1"])
    pred = stats.linregress(dp["signed_vol"].values / scale, dp["fwd1"].values)

    # Amihud illiquidity: |return| vs volume
    d2 = df.dropna(subset=["logret", "volume"])
    amihud = stats.linregress(d2["volume"].values, d2["logret"].abs().values)
    return {
        "kyle_contemp_r2": float(contemp.rvalue ** 2),           # inflated by construction
        "kyle_pred_lambda": float(pred.slope), "kyle_pred_p": float(pred.pvalue),
        "kyle_pred_r2": float(pred.rvalue ** 2),                 # the honest number
        "amihud_slope": float(amihud.slope), "amihud_p": float(amihud.pvalue),
    }


def cusum_events(series, h=5.0):
    """Symmetric CUSUM filter (López de Prado): flags when CUMULATIVE standardized
    flow crosses ±h — i.e. the point at which flow stops being noise. Returns indices."""
    s = np.nan_to_num(np.asarray(series, dtype=float))
    events, sp, sn = [], 0.0, 0.0
    for i, x in enumerate(s):
        sp = max(0.0, sp + x)
        sn = min(0.0, sn + x)
        if sp > h:
            sp = 0.0; events.append(i)
        elif sn < -h:
            sn = 0.0; events.append(i)
    return events


def significant_threshold(df, move=0.02, horizon=1):
    """Logistic: does a flow anomaly raise P(|move| >= `move` over `horizon` days)?
    The answer *is* the significant-threshold relationship, expressed as a probability."""
    from sklearn.linear_model import LogisticRegression
    d = df.dropna(subset=["vol_z", "signed_vol_z", "logret"]).copy()
    fwd = np.log(d["close"]).shift(-horizon) - np.log(d["close"])
    d["big_move"] = (fwd.abs() >= np.log(1 + move)).astype(int)
    d = d.dropna(subset=["big_move"])
    X = d[["vol_z", "signed_vol_z"]].values
    y = d["big_move"].values
    if y.sum() < 10:
        return {"error": "too few big-move events"}
    # no class balancing here: we want interpretable probabilities near the base rate
    with np.errstate(all="ignore"):          # benign transient LBFGS FP warnings
        m = LogisticRegression(max_iter=1000).fit(X, y)
    # probability at a calm day (z=0) vs a strong-flow day (z=+3)
    p_calm = m.predict_proba([[0, 0]])[0, 1]
    p_flow = m.predict_proba([[3, 3]])[0, 1]
    return {
        "base_rate": round(float(y.mean()), 3),
        "coef_vol_z": round(float(m.coef_[0][0]), 4),
        "coef_signed_z": round(float(m.coef_[0][1]), 4),
        "P_big_move_calm": round(float(p_calm), 3),
        "P_big_move_high_flow": round(float(p_flow), 3),
    }


if __name__ == "__main__":
    print("=== LIVE options net-premium flow ===")
    lf = live_option_flow("BTC")
    print(f"  {lf['n_trades']} trades · net call {lf['net_call_premium']:+.2f} "
          f"put {lf['net_put_premium']:+.2f} BTC · tilt: {lf['flow_tilt']}")

    print("\n=== SIGNIFICANT FLOW THRESHOLD (2y daily) ===")
    df = pd.read_csv("data/market_daily.csv", parse_dates=["date"])
    df = add_flow_features(df)

    kl = kyle_lambda(df)
    star = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "(ns)"
    print(f"  Kyle λ PREDICTIVE (flow->next ret) = {kl['kyle_pred_lambda']:+.5f}  "
          f"p={kl['kyle_pred_p']:.4f} {star(kl['kyle_pred_p'])}  R²={kl['kyle_pred_r2']:.3f}  (honest)")
    print(f"  (contemporaneous R²={kl['kyle_contemp_r2']:.2f} — inflated by construction, ignored)")
    print(f"  Amihud illiquidity  = {kl['amihud_slope']:+.2e}  p={kl['amihud_p']:.4f} {star(kl['amihud_p'])}")

    ev = cusum_events(df["signed_vol_z"], h=5.0)
    print(f"  CUSUM significant-flow events (h=5): {len(ev)}")

    st_ = significant_threshold(df)
    if "error" not in st_:
        print(f"  P(|move|>=2% in 1d):  calm day = {st_['P_big_move_calm']}  "
              f"vs high-flow day = {st_['P_big_move_high_flow']}   (base {st_['base_rate']})")
        print(f"    coefficients: vol_z={st_['coef_vol_z']}  signed_vol_z={st_['coef_signed_z']}")
