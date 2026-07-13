"""
stats_spine.py — The statistical evaluation layer (Python side).

This is the core of the project: it takes the anomalies/regimes and asks, rigorously,
whether the article's claims survive contact with the data. Every claim becomes a test.

  1. event_study()        CAR around DVOL-spike events (mean-adjusted abnormal returns)
  2. regime_two_sample()  forward realized vol & return: elevated vs calm regime
  3. return_persistence() lag-1 autocorrelation by regime (trending vs mean-reverting)
  4. permutation_test()   distribution-free robustness check
  5. logistic_probability() calibrated P(up-move | regime features) + Brier/reliability

Honesty notes are printed inline. Overlapping forward windows mean p-values are
optimistic; a purged/embargoed CV (Phase 2) tightens this — flagged where relevant.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from anomalies import add_forward_returns, add_regime_labels, detect_dvol_spikes

RNG = np.random.default_rng(42)


def event_study(df, events, ret_col="logret", pre=3, post=5):
    """Mean-adjusted CAR around events. Returns per-offset mean AR, CAR, and t-stats."""
    base = df[ret_col].mean()
    offsets = range(-pre, post + 1)
    mat = []
    for i in events:
        if i - pre < 0 or i + post >= len(df):
            continue
        ar = df[ret_col].iloc[i - pre:i + post + 1].values - base
        mat.append(ar)
    mat = np.array(mat)
    if len(mat) == 0:
        return None
    mean_ar = mat.mean(axis=0)
    se = mat.std(axis=0, ddof=1) / np.sqrt(len(mat))
    tstat = mean_ar / se
    car = np.cumsum(mean_ar)
    return pd.DataFrame({"offset": list(offsets), "mean_AR": mean_ar,
                         "t_AR": tstat, "CAR": car}), len(mat)


def regime_two_sample(df, value_col, regime_col="dvol_regime"):
    """Compare a value across 'elevated' vs 'calm' regimes with 3 complementary tests."""
    a = df.loc[df[regime_col] == "elevated", value_col].dropna()
    b = df.loc[df[regime_col] == "calm", value_col].dropna()
    welch = stats.ttest_ind(a, b, equal_var=False)
    mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
    levene = stats.levene(a, b)  # variance equality (the vol-clustering claim)
    return {
        "n_elevated": len(a), "n_calm": len(b),
        "mean_elevated": float(a.mean()), "mean_calm": float(b.mean()),
        "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
        "mwu_p": float(mwu.pvalue),
        "levene_p": float(levene.pvalue),
    }


def return_persistence(df, regime_col="dvol_regime", ret_col="logret"):
    """Lag-1 autocorrelation of returns by regime.

    Article claim: negative-gamma (proxy: elevated vol) -> trending (autocorr>0);
    positive-gamma (proxy: calm) -> mean-reverting/sticky (autocorr<=0).
    """
    out = {}
    for reg in ("elevated", "calm"):
        r = df.loc[df[regime_col] == reg, ret_col].dropna()
        ac1 = r.autocorr(lag=1)
        # significance of autocorr ~ N(0, 1/n)
        z = ac1 * np.sqrt(len(r))
        out[reg] = {"autocorr_lag1": float(ac1), "n": len(r), "z": float(z),
                    "p": float(2 * (1 - stats.norm.cdf(abs(z))))}
    return out


def permutation_test(a, b, n_perm=10000):
    """Distribution-free test that mean(a) != mean(b)."""
    a, b = np.asarray(a), np.asarray(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na = len(a)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        p = RNG.permutation(pool)
        diffs[i] = p[:na].mean() - p[na:].mean()
    p = float((np.abs(diffs) >= abs(obs)).mean())
    return {"observed_diff": float(obs), "p_value": p}


def logistic_probability(df, target_col, feature_cols, up_thresh=0.02):
    """Calibrated P(up-move >= up_thresh over the target horizon | regime features)."""
    d = df.dropna(subset=[target_col] + feature_cols).copy()
    d["y"] = (d[target_col] >= np.log(1 + up_thresh)).astype(int)
    X = d[feature_cols].values
    y = d["y"].values
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return {"error": "too few events for a stable fit"}

    # time-ordered split (honest-ish; purged CV comes in Phase 2)
    split = int(len(d) * 0.7)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    calib = CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(3))
    with np.errstate(all="ignore"):          # benign transient LBFGS FP warnings
        base.fit(Xtr, ytr)
        calib.fit(Xtr, ytr)
        p_te = calib.predict_proba(Xte)[:, 1]
    brier = brier_score_loss(yte, p_te)
    base_rate = float(y.mean())
    # reliability (few bins given small data)
    try:
        frac_pos, mean_pred = calibration_curve(yte, p_te, n_bins=5, strategy="quantile")
        reliab = list(zip([round(x, 3) for x in mean_pred], [round(x, 3) for x in frac_pos]))
    except Exception:
        reliab = []
    # coefficients on standardized features (comparable => usable as factor weights)
    lr = base.named_steps["logisticregression"]
    coefs = dict(zip(feature_cols, [round(float(c), 4) for c in lr.coef_[0]]))
    return {
        "base_rate": round(base_rate, 3),
        "brier": round(float(brier), 4),
        "coefficients": coefs,
        "reliability_pred_vs_obs": reliab,
        "n_train": len(ytr), "n_test": len(yte),
    }


def purged_walk_forward(df, target_col, feature_cols, horizon=3, up_thresh=0.02,
                        n_folds=5):
    """Leakage-safe out-of-sample probability via purged, embargoed walk-forward.

    Between each train block and its test block we drop `horizon` days (the embargo)
    so a label whose forward window overlaps the test set can't leak into training.
    Returns Brier on the concatenated OOS predictions vs the base-rate benchmark —
    the honest measure of whether the model actually adds skill.
    """
    d = df.dropna(subset=[target_col] + feature_cols).reset_index(drop=True)
    y = (d[target_col] >= np.log(1 + up_thresh)).astype(int).values
    X = d[feature_cols].values
    n = len(d)
    fold = n // (n_folds + 1)
    oos_p, oos_y = [], []
    for k in range(1, n_folds + 1):
        tr_end = k * fold - horizon               # embargo: purge overlap
        te_a, te_b = k * fold, min((k + 1) * fold, n)
        if tr_end < 30 or te_b <= te_a:
            continue
        ytr = y[:tr_end]
        if ytr.sum() < 5 or (len(ytr) - ytr.sum()) < 5:
            continue
        # scaled + unbalanced: numerically stable, true-frequency probs for a fair Brier
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        with np.errstate(all="ignore"):      # benign transient LBFGS FP warnings
            m.fit(X[:tr_end], ytr)           # scaler fits on train only — no leakage
            oos_p.extend(m.predict_proba(X[te_a:te_b])[:, 1])
        oos_y.extend(y[te_a:te_b])
    if not oos_p:
        return {"error": "insufficient data for purged CV"}
    oos_p, oos_y = np.array(oos_p), np.array(oos_y)
    base = oos_y.mean()
    return {
        "n_oos": len(oos_y),
        "brier_model": round(float(brier_score_loss(oos_y, oos_p)), 4),
        "brier_baserate": round(float(brier_score_loss(oos_y, np.full_like(oos_p, base))), 4),
        "base_rate": round(float(base), 3),
    }


def garch_persistence(df, ret_col="logret"):
    """GARCH(1,1) volatility clustering — the test we skipped when R's rugarch
    wouldn't install. `arch` does it natively in Python. persistence = alpha+beta;
    > 0.9 means strong, long-memory clustering (the article's regime thesis)."""
    from arch import arch_model
    r = df[ret_col].dropna().values * 100
    res = arch_model(r, vol="Garch", p=1, q=1, dist="t", mean="Constant").fit(disp="off")
    a = float(res.params.get("alpha[1]", float("nan")))
    b = float(res.params.get("beta[1]", float("nan")))
    return {"alpha": round(a, 4), "beta": round(b, 4),
            "persistence": round(a + b, 4), "strong_clustering": (a + b) > 0.9}


def _p(x):
    return f"{x:.4f}" + ("  ***" if x < 0.01 else "  **" if x < 0.05 else "  (ns)")


def main():
    df = pd.read_csv("data/market_daily.csv", parse_dates=["date"])
    df = add_forward_returns(df, horizons=(1, 3, 7))
    df = add_regime_labels(df)
    spikes = detect_dvol_spikes(df, z_thresh=2.0)

    print("=" * 66)
    print("  STATISTICAL EVALUATION — dealer-positioning thesis vs BTC data")
    print(f"  {len(df)} days  |  {df['date'].min().date()} -> {df['date'].max().date()}")
    print("=" * 66)

    print("\n[1] EVENT STUDY — DVOL spikes (|z|>2), mean-adjusted CAR")
    es = event_study(df, spikes, pre=3, post=5)
    if es:
        tab, n = es
        print(f"    events used: {n}")
        print(tab.round(4).to_string(index=False))

    print("\n[2] REGIME TWO-SAMPLE — elevated vs calm implied-vol regime")
    for col, claim in [("fwd_rv_3", "forward realized vol (vol clustering)"),
                       ("fwd_ret_3", "forward 3d return (directional bias)")]:
        r = regime_two_sample(df, col)
        print(f"  · {claim}")
        print(f"      mean elevated={r['mean_elevated']:+.4f}  calm={r['mean_calm']:+.4f}"
              f"   (n {r['n_elevated']}/{r['n_calm']})")
        print(f"      Welch p={_p(r['welch_p'])}   MWU p={_p(r['mwu_p'])}   Levene(var) p={_p(r['levene_p'])}")

    print("\n[3] RETURN PERSISTENCE — lag-1 autocorrelation by regime")
    rp = return_persistence(df)
    for reg, v in rp.items():
        tag = "trending" if v["autocorr_lag1"] > 0 else "mean-reverting"
        print(f"    {reg:>8}: autocorr={v['autocorr_lag1']:+.3f}  ({tag})  p={_p(v['p'])}  n={v['n']}")

    print("\n[4] PERMUTATION — forward 3d realized vol, elevated vs calm")
    a = df.loc[df.dvol_regime == "elevated", "fwd_rv_3"].dropna()
    b = df.loc[df.dvol_regime == "calm", "fwd_rv_3"].dropna()
    pt = permutation_test(a, b)
    print(f"    observed diff={pt['observed_diff']:+.3f}   p={_p(pt['p_value'])}")

    print("\n[5] CALIBRATED PROBABILITY — P(+2% over 3d | regime features)")
    df["elevated"] = (df["dvol_regime"] == "elevated").astype(int)
    lp = logistic_probability(df, "fwd_ret_3", ["elevated", "dvol_chg_z", "dvol"])
    if "error" in lp:
        print("    " + lp["error"])
    else:
        print(f"    base rate={lp['base_rate']}  Brier={lp['brier']}  (train {lp['n_train']} / test {lp['n_test']})")
        print(f"    coefficients: {lp['coefficients']}")
        print(f"    reliability (pred vs observed): {lp['reliability_pred_vs_obs']}")

    print("\n[G] GARCH(1,1) — volatility clustering (Python arch, replaces rugarch)")
    try:
        g = garch_persistence(df)
        tag = "strong long-memory clustering" if g["strong_clustering"] else "weak"
        print(f"    alpha={g['alpha']}  beta={g['beta']}  persistence={g['persistence']}  -> {tag}")
    except Exception as e:
        print(f"    garch unavailable: {e}")

    print("\n[6] PURGED WALK-FORWARD — leakage-safe skill check (Brier)")
    pw = purged_walk_forward(df, "fwd_ret_3", ["elevated", "dvol_chg_z", "dvol"])
    if "error" in pw:
        print("    " + pw["error"])
    else:
        edge = pw["brier_baserate"] - pw["brier_model"]
        if edge <= 0:
            verdict = "NO skill over base rate (honest null)"
        elif edge < 0.01:
            verdict = "negligible edge (within noise)"
        else:
            verdict = "modest edge over base rate"
        print(f"    OOS n={pw['n_oos']}  Brier model={pw['brier_model']}  "
              f"base-rate={pw['brier_baserate']}  edge={edge:+.4f}  -> {verdict}")

    print("\n" + "-" * 66)
    print("  NOTE: DVOL regime is an observable *proxy* for the dealer gamma regime.")
    print("  The GEX-native test matures as the forward snapshot DB grows.")
    print("  Overlapping windows => p-values optimistic; purged CV in Phase 2.")


if __name__ == "__main__":
    main()
