"""
stats_spine.py — The statistical evaluation layer (Python side).

Takes the anomalies/regimes and asks whether the positioning thesis's claims
survive contact with the data. Every claim becomes a test:

  1. event_study()          CAR around DVOL-spike events, benchmarked against a
                            pre-event estimation window
  2. regime_two_sample()    forward realized vol & return: elevated vs calm regime
  3. return_persistence()   lag-1 autocorrelation by regime, contiguous pairs only
  4. permutation_test()     distribution-free robustness check
  5. logistic_probability() calibrated P(up-move | regime features) on a single
                            time-ordered split (diagnostic, not the headline)
  6. purged_walk_forward()  expanding-window walk-forward with a horizon-length
                            gap before each test block; naive_walk_forward() is
                            the same thing with no gap, printed side by side

All regime-conditioned tests use the same sample: rows with a valid regime
label (DVOL present and a trailing median available). Overlapping forward
windows make the in-sample p-values optimistic; the walk-forward Brier is the
number to trust for anything predictive.
"""

import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anomalies import add_forward_returns, add_regime_labels, detect_dvol_spikes

RNG = np.random.default_rng(42)
DATA_FILE = "data/market_daily.csv"
REGIME_FEATURES = ["elevated", "dvol_chg_z", "dvol"]


def event_study(df, events, ret_col="logret", pre=3, post=5, est_window=60):
    """Abnormal returns around events, benchmarked against a pre-event window.

    For each event at index i the expected return is the mean of `ret_col` over
    the `est_window` days ending at i - pre - 1 (so the estimation window never
    touches the event window). Events without a full estimation window or a
    full event window are skipped. Event windows are allowed to overlap, which
    makes the t-statistics optimistic when events cluster — see docs.

    Returns (table, n_events_used) or None if no event qualifies.
    """
    r = df[ret_col].values
    offsets = range(-pre, post + 1)
    mat = []
    for i in events:
        est_a, est_b = i - pre - est_window, i - pre        # [est_a, est_b)
        if est_a < 0 or i + post >= len(df):
            continue
        est = r[est_a:est_b]
        est = est[~np.isnan(est)]
        if len(est) < est_window // 2:
            continue
        ar = r[i - pre:i + post + 1] - est.mean()
        if np.isnan(ar).any():
            continue
        mat.append(ar)
    if not mat:
        return None
    mat = np.array(mat)
    mean_ar = mat.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):   # a single event has no t-stat
        se = mat.std(axis=0, ddof=1) / np.sqrt(len(mat))
        tstat = mean_ar / se
    car = np.cumsum(mean_ar)
    return pd.DataFrame({"offset": list(offsets), "mean_AR": mean_ar,
                         "t_AR": tstat, "CAR": car}), len(mat)


def regime_two_sample(df, value_col, regime_col="dvol_regime"):
    """Compare a value across 'elevated' vs 'calm' regimes with 3 complementary tests.

    Rows whose regime is NaN (no DVOL) belong to neither sample.
    """
    a = df.loc[df[regime_col] == "elevated", value_col].dropna()
    b = df.loc[df[regime_col] == "calm", value_col].dropna()
    welch = stats.ttest_ind(a, b, equal_var=False)
    mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
    levene = stats.levene(a, b)  # variance equality
    return {
        "n_elevated": len(a), "n_calm": len(b),
        "mean_elevated": float(a.mean()), "mean_calm": float(b.mean()),
        "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
        "mwu_p": float(mwu.pvalue),
        "levene_p": float(levene.pvalue),
    }


def return_persistence(df, regime_col="dvol_regime", ret_col="logret"):
    """Lag-1 autocorrelation of returns by regime, using contiguous pairs only.

    A pair (r[t-1], r[t]) counts for a regime only if both days carry that
    regime label, so a calm day never gets paired with the elevated day that
    preceded it. Claim under test: elevated (negative-gamma proxy) -> trending
    (autocorr > 0); calm (positive-gamma proxy) -> mean-reverting (autocorr < 0).
    """
    r = df[ret_col]
    reg = df[regime_col]
    out = {}
    for name in ("elevated", "calm"):
        same = (reg == name) & (reg.shift(1) == name) & r.notna() & r.shift(1).notna()
        x, y = r.shift(1)[same].values, r[same].values
        n = len(x)
        ac1 = float(np.corrcoef(x, y)[0, 1]) if n > 2 else float("nan")
        z = ac1 * np.sqrt(n)                       # autocorr ~ N(0, 1/n) under H0
        out[name] = {"autocorr_lag1": ac1, "n": int(n), "z": float(z),
                     "p": float(2 * (1 - stats.norm.cdf(abs(z))))}
    return out


def permutation_test(a, b, n_perm=10000):
    """Distribution-free test that mean(a) != mean(b)."""
    a, b = np.asarray(a), np.asarray(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        p = RNG.permutation(pool)
        diffs[i] = p[:na].mean() - p[na:].mean()
    p = float((np.abs(diffs) >= abs(obs)).mean())
    return {"observed_diff": float(obs), "p_value": p}


def logistic_probability(df, target_col, feature_cols, up_thresh=0.02):
    """Calibrated P(up-move >= up_thresh | features) on ONE time-ordered 70/30 split.

    Diagnostic only: it shows what sigmoid calibration does to the reliability
    curve. The headline skill number is purged_walk_forward(), which uses an
    uncalibrated LogisticRegression.
    """
    d = df.dropna(subset=[target_col] + feature_cols).copy()
    d["y"] = (d[target_col] >= np.log(1 + up_thresh)).astype(int)
    X = d[feature_cols].values
    y = d["y"].values
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return {"error": "too few events for a stable fit"}

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
    try:
        frac_pos, mean_pred = calibration_curve(yte, p_te, n_bins=5, strategy="quantile")
        reliab = [(round(float(m), 3), round(float(f), 3)) for m, f in zip(mean_pred, frac_pos)]
    except ValueError:
        reliab = []
    lr = base.named_steps["logisticregression"]
    coefs = dict(zip(feature_cols, [round(float(c), 4) for c in lr.coef_[0]]))
    return {
        "base_rate": round(base_rate, 3),
        "brier": round(float(brier), 4),
        "coefficients": coefs,
        "reliability_pred_vs_obs": reliab,
        "n_train": len(ytr), "n_test": len(yte),
    }


# ----------------------------------------------------------------------------
# Walk-forward validation
# ----------------------------------------------------------------------------
def horizon_from_target(target_col):
    """'fwd_ret_3' -> 3. Raises ValueError if the name carries no trailing integer."""
    m = re.search(r"(\d+)$", str(target_col))
    if not m:
        raise ValueError(f"cannot infer horizon from target column {target_col!r}")
    return int(m.group(1))


def walk_forward_splits(n, n_folds, horizon, gap=None):
    """Expanding-window walk-forward split indices.

    The sample is cut into n_folds + 1 equal blocks. Fold k trains on everything
    before block k minus the last `gap` rows, and tests on block k. With
    gap = horizon (the default) no training label's forward window reaches into
    the test block:  max(train) + horizon < min(test).  gap = 0 is the naive
    split that lets overlapping labels straddle the boundary.

    Yields (train_idx, test_idx) as integer arrays.
    """
    gap = horizon if gap is None else gap
    fold = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        te_a, te_b = k * fold, min((k + 1) * fold, n)
        tr_end = te_a - gap
        if tr_end <= 0 or te_b <= te_a:
            continue
        yield np.arange(0, tr_end), np.arange(te_a, te_b)


def _default_model():
    # scaled + unbalanced: numerically stable, true-frequency probabilities
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def walk_forward_brier(X, y, horizon, gap, n_folds=5, model_factory=_default_model,
                       min_train=30, min_class=5):
    """Out-of-sample Brier from concatenated walk-forward predictions.

    The model (and any scaler inside it) is fitted on the training slice only.
    Returns Brier for the model and for a base-rate predictor on the same rows.
    """
    X, y = np.asarray(X), np.asarray(y)
    oos_p, oos_y = [], []
    for tr, te in walk_forward_splits(len(y), n_folds, horizon, gap):
        ytr = y[tr]
        if len(tr) < min_train or ytr.sum() < min_class or (len(ytr) - ytr.sum()) < min_class:
            continue
        m = model_factory()
        with np.errstate(all="ignore"):      # benign transient LBFGS FP warnings
            m.fit(X[tr], ytr)
            oos_p.extend(m.predict_proba(X[te])[:, 1])
        oos_y.extend(y[te])
    if not oos_p:
        return {"error": "insufficient data for walk-forward validation"}
    oos_p, oos_y = np.array(oos_p), np.array(oos_y)
    base = oos_y.mean()
    return {
        "n_oos": int(len(oos_y)),
        "gap": int(gap),
        "brier_model": round(float(brier_score_loss(oos_y, oos_p)), 4),
        "brier_baserate": round(float(brier_score_loss(oos_y, np.full_like(oos_p, base))), 4),
        "base_rate": round(float(base), 3),
    }


def _design(df, target_col, feature_cols, up_thresh):
    cols = [target_col] + list(feature_cols)
    if "dvol_regime" in df.columns:
        cols.append("dvol_regime")              # regime tests share one sample
    d = df.dropna(subset=cols).reset_index(drop=True)
    y = (d[target_col] >= np.log(1 + up_thresh)).astype(int).values
    return d[list(feature_cols)].values, y


def purged_walk_forward(df, target_col, feature_cols, horizon=None, up_thresh=0.02,
                        n_folds=5):
    """Walk-forward Brier with a horizon-length gap before each test block.

    The horizon is read from the target name (fwd_ret_3 -> 3). Passing an
    explicit `horizon` that disagrees with the target raises ValueError, so the
    gap can never silently be shorter than the label window.
    """
    h = horizon_from_target(target_col)
    if horizon is not None and horizon != h:
        raise ValueError(f"horizon={horizon} but target {target_col!r} implies {h}")
    X, y = _design(df, target_col, feature_cols, up_thresh)
    return walk_forward_brier(X, y, horizon=h, gap=h, n_folds=n_folds)


def naive_walk_forward(df, target_col, feature_cols, up_thresh=0.02, n_folds=5):
    """Same walk-forward, no gap: overlapping labels straddle train/test boundaries."""
    h = horizon_from_target(target_col)
    X, y = _design(df, target_col, feature_cols, up_thresh)
    return walk_forward_brier(X, y, horizon=h, gap=0, n_folds=n_folds)


def garch_persistence(df, ret_col="logret"):
    """GARCH(1,1) via the `arch` package. persistence = alpha + beta;
    > 0.9 means strong, long-memory volatility clustering."""
    from arch import arch_model
    r = df[ret_col].dropna().values * 100
    res = arch_model(r, vol="Garch", p=1, q=1, dist="t", mean="Constant").fit(disp="off")
    a = float(res.params.get("alpha[1]", float("nan")))
    b = float(res.params.get("beta[1]", float("nan")))
    return {"alpha": round(a, 4), "beta": round(b, 4),
            "persistence": round(a + b, 4), "strong_clustering": (a + b) > 0.9}


def _p(x):
    return f"{x:.4f}" + ("  ***" if x < 0.01 else "  **" if x < 0.05 else "  (ns)")


def load(path=DATA_FILE):
    df = pd.read_csv(path, parse_dates=["date"])
    df = add_forward_returns(df, horizons=(1, 3, 7))
    return add_regime_labels(df)


def main(path=DATA_FILE):
    df = load(path)
    spikes = detect_dvol_spikes(df, z_thresh=2.0)
    labelled = df["dvol_regime"].notna()

    print("=" * 66)
    print("  STATISTICAL EVALUATION — dealer-positioning thesis vs BTC data")
    print(f"  {len(df)} days  |  {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  regime-labelled days: {int(labelled.sum())}  "
          f"({df.loc[labelled, 'date'].min().date()} -> {df.loc[labelled, 'date'].max().date()}; "
          f"DVOL missing on {int(df['dvol'].isna().sum())} days, 19 warm-up days)")
    print("=" * 66)

    print("\n[1] EVENT STUDY — DVOL spikes (|z|>2), AR vs 60-day pre-event mean")
    es = event_study(df, spikes, pre=3, post=5)
    if es:
        tab, n = es
        print(f"    events used: {n} of {len(spikes)} (windows may overlap)")
        print(tab.round(4).to_string(index=False))

    print("\n[2] REGIME TWO-SAMPLE — elevated vs calm implied-vol regime")
    for col, claim in [("fwd_rv_3", "forward 3d realized vol (vol clustering)"),
                       ("fwd_ret_3", "forward 3d return (directional bias)")]:
        r = regime_two_sample(df, col)
        print(f"  · {claim}")
        print(f"      mean elevated={r['mean_elevated']:+.4f}  calm={r['mean_calm']:+.4f}"
              f"   (n {r['n_elevated']}/{r['n_calm']})")
        print(f"      Welch p={_p(r['welch_p'])}   MWU p={_p(r['mwu_p'])}   Levene(var) p={_p(r['levene_p'])}")

    print("\n[3] RETURN PERSISTENCE — lag-1 autocorrelation, contiguous same-regime pairs")
    rp = return_persistence(df)
    for reg, v in rp.items():
        tag = "trending" if v["autocorr_lag1"] > 0 else "mean-reverting"
        print(f"    {reg:>8}: autocorr={v['autocorr_lag1']:+.3f}  ({tag})  p={_p(v['p'])}  pairs={v['n']}")

    print("\n[4] PERMUTATION — forward 3d realized vol, elevated vs calm")
    a = df.loc[df.dvol_regime == "elevated", "fwd_rv_3"].dropna()
    b = df.loc[df.dvol_regime == "calm", "fwd_rv_3"].dropna()
    pt = permutation_test(a, b)
    print(f"    observed diff={pt['observed_diff']:+.3f}   p={_p(pt['p_value'])}")

    print("\n[5] CALIBRATED PROBABILITY — P(+2% over 3d | regime features), single 70/30 time split")
    lp = logistic_probability(df, "fwd_ret_3", REGIME_FEATURES)
    if "error" in lp:
        print("    " + lp["error"])
    else:
        print(f"    base rate={lp['base_rate']}  Brier={lp['brier']}  (train {lp['n_train']} / test {lp['n_test']})")
        print(f"    coefficients: {lp['coefficients']}")
        print(f"    reliability (pred vs observed): {lp['reliability_pred_vs_obs']}")

    print("\n[G] GARCH(1,1) — volatility clustering (Python arch)")
    try:
        g = garch_persistence(df)
        tag = "strong long-memory clustering" if g["strong_clustering"] else "weak"
        print(f"    alpha={g['alpha']}  beta={g['beta']}  persistence={g['persistence']}  -> {tag}")
    except (ImportError, ValueError) as e:
        print(f"    garch unavailable: {e}")

    print("\n[6] WALK-FORWARD SKILL — uncalibrated LogisticRegression, 5 expanding folds")
    pw = purged_walk_forward(df, "fwd_ret_3", REGIME_FEATURES)
    nw = naive_walk_forward(df, "fwd_ret_3", REGIME_FEATURES)
    for label, r in (("purged (gap=3)", pw), ("naive  (gap=0)", nw)):
        if "error" in r:
            print(f"    {label}: {r['error']}")
            continue
        edge = r["brier_baserate"] - r["brier_model"]
        if edge <= 0:
            verdict = "no skill over base rate"
        elif edge < 0.01:
            verdict = "negligible edge (within noise)"
        else:
            verdict = "modest edge over base rate"
        print(f"    {label}: OOS n={r['n_oos']}  Brier model={r['brier_model']}  "
              f"base-rate={r['brier_baserate']}  edge={edge:+.4f}  -> {verdict}")

    print("\n" + "-" * 66)
    print("  NOTE: DVOL regime is an observable *proxy* for the dealer gamma regime.")
    print("  Regime tests cover only the DVOL-labelled days, not the full 731.")
    print("  Overlapping forward windows => in-sample p-values optimistic.")


if __name__ == "__main__":
    main()
