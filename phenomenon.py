"""
phenomenon.py — A general engine to statistically analyze ANY market phenomenon.

You define a phenomenon as a boolean condition (the "event"); the engine runs the
full battery automatically and returns an honest verdict:

  · event study        — cumulative abnormal return (CAR) around the event
  · forward return      — event days vs the rest (Welch, Mann-Whitney, permutation)
  · forward volatility  — event days vs the rest (Levene: does the event change vol?)
  · big-move odds       — P(|move| >= X%) on event days vs baseline (two-proportion z)

This is what turns the project from "6 fixed tests" into a reusable instrument:
point it at a condition, get a rigorous read on whether it matters.

Example:
    df = load()
    analyze_phenomenon(df, df["dvol_chg_z"] > 2, "DVOL spike")
    analyze_phenomenon(df, df["logret"] < -0.03, "3%+ down day")
"""

import numpy as np
import pandas as pd
from scipy import stats

from anomalies import add_forward_returns, add_regime_labels
from flow_engine import add_flow_features
from stats_spine import event_study, permutation_test


def _two_proportion_z(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def analyze_phenomenon(df, event_mask, label, ret_col="fwd_ret_3",
                       vol_col="fwd_rv_3", up_thresh=0.02, verbose=True):
    """Run the full statistical battery on one phenomenon. Returns a result dict."""
    df = df.reset_index(drop=True)
    mask = pd.Series(event_mask).reset_index(drop=True).fillna(False).astype(bool)
    ev_idx = df.index[mask].tolist()

    res = {"label": label, "n_events": int(mask.sum()), "n_total": len(df)}

    # 1) event study
    es = event_study(df, ev_idx, ret_col="logret", pre=3, post=5)
    if es:
        tab, n = es
        res["event_study"] = {"n": n, "CAR_end": round(float(tab["CAR"].iloc[-1]), 4),
                              "max_abs_t": round(float(tab["t_AR"].abs().max()), 2)}

    # 2) forward return: event vs rest
    a = df.loc[mask, ret_col].dropna()
    b = df.loc[~mask, ret_col].dropna()
    if len(a) > 5 and len(b) > 5:
        res["fwd_return"] = {
            "mean_event": round(float(a.mean()), 4), "mean_rest": round(float(b.mean()), 4),
            "welch_p": round(float(stats.ttest_ind(a, b, equal_var=False).pvalue), 4),
            "mwu_p": round(float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue), 4),
            "perm_p": round(permutation_test(a.values, b.values, 5000)["p_value"], 4),
        }

    # 3) forward volatility: does the event change vol?
    av = df.loc[mask, vol_col].dropna()
    bv = df.loc[~mask, vol_col].dropna()
    if len(av) > 5 and len(bv) > 5:
        res["fwd_vol"] = {
            "mean_event": round(float(av.mean()), 2), "mean_rest": round(float(bv.mean()), 2),
            "welch_p": round(float(stats.ttest_ind(av, bv, equal_var=False).pvalue), 4),
            "levene_p": round(float(stats.levene(av, bv).pvalue), 4),
        }

    # 4) big-move odds: P(|move| >= up_thresh)
    big = df[ret_col].abs() >= np.log(1 + up_thresh)
    k1, n1 = int(big[mask].sum()), int(mask.sum())
    k2, n2 = int(big[~mask].sum()), int((~mask).sum())
    z, pz = _two_proportion_z(k1, n1, k2, n2)
    res["big_move"] = {"p_event": round(k1 / n1, 3) if n1 else None,
                       "p_rest": round(k2 / n2, 3) if n2 else None,
                       "z_p": round(float(pz), 4) if not np.isnan(pz) else None}

    if verbose:
        _print(res)
    return res


def _sig(p):
    return "***" if p is not None and p < 0.01 else "**" if p is not None and p < 0.05 else "(ns)"


def _print(r):
    print(f"\n{'='*60}\n  PHENOMENON: {r['label']}   ({r['n_events']}/{r['n_total']} days)\n{'='*60}")
    if "event_study" in r:
        e = r["event_study"]
        print(f"  Event study : CAR(end)={e['CAR_end']:+.3f}  max|t|={e['max_abs_t']}")
    if "fwd_return" in r:
        f = r["fwd_return"]
        print(f"  Fwd return  : event={f['mean_event']:+.4f} vs rest={f['mean_rest']:+.4f}"
              f"   perm p={f['perm_p']} {_sig(f['perm_p'])}")
    if "fwd_vol" in r:
        v = r["fwd_vol"]
        print(f"  Fwd vol     : event={v['mean_event']:.1f}% vs rest={v['mean_rest']:.1f}%"
              f"   Welch p={v['welch_p']} {_sig(v['welch_p'])}")
    if "big_move" in r:
        b = r["big_move"]
        print(f"  Big-move Pr : event={b['p_event']} vs rest={b['p_rest']}"
              f"   z p={b['z_p']} {_sig(b['z_p'])}")


def load():
    df = pd.read_csv("data/market_daily.csv", parse_dates=["date"])
    return add_flow_features(add_regime_labels(add_forward_returns(df)))


if __name__ == "__main__":
    df = load()
    # three example phenomena — each is just a boolean condition
    analyze_phenomenon(df, df["dvol_chg_z"] > 2, "DVOL spike (implied-vol jump)")
    analyze_phenomenon(df, df["logret"] < -0.03, "3%+ down day (capitulation?)")
    analyze_phenomenon(df, df["vol_z"] > 2, "High-volume day")
    print("\n-> Define any boolean condition on the data and it runs this automatically.")
