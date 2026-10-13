# Methodology

A technical explainer of what the project tests and how the validation is set
up. Every number here is produced by `python3 stats_spine.py`,
`python3 flow_engine.py` or `Rscript inference.R` on the frozen sample.

## Hypothesis

The market-microstructure claim under test (a popular retail
options-positioning thesis):

> Dealers who hedge their options-book exposure create observable pressure on
> the underlying market. When aggregate gamma exposure is *positive* they
> hedge counter-trend and the tape becomes calmer and mean-reverting; when it
> is *negative* they hedge with-trend and the tape trends. Large flow events
> raise the odds of a large price move.

It predicts differences in **forward realized volatility**, **return
autocorrelation**, and **the probability of a large move** conditional on the
regime and on flow.

## Data

- **Source:** Deribit public REST — free, no API key.
- **Frequency:** daily bars, 2024-07-13 → 2026-07-13, **731 rows**
  (`data/sample_2024-07_2026-07.csv`, a frozen copy of `market_daily.csv`).
- **Fields:** OHLCV on BTC-PERPETUAL (volume in BTC), Deribit's DVOL
  implied-volatility index, and the live option chain for positioning.
- **Regime proxy:** DVOL relative to its trailing 90-day median
  (`min_periods = 20`) labels each day `elevated` or `calm`. Days with no
  DVOL, or no median yet, get **no label**. DVOL is present for 500 of 731
  days, so **481 days (2025-03-20 → 2026-07-13) carry a regime label** and
  every regime-conditioned test uses exactly that sample.

## Targets

For each day *t*:

- `fwd_ret_h` — log return from *t* to *t + h*, `h ∈ {1, 3, 7}`.
- `fwd_rv_h` — annualised realized volatility of returns *t + 1 … t + h*,
  `h ∈ {3, 7}` (a one-day window has no standard deviation).
- Binary "up move": `fwd_ret_3 ≥ log(1.02)`; binary "big move":
  `|fwd_ret_1| ≥ log(1.02)`.

## Tests and results

| Question | Test | Result (frozen sample) |
|---|---|---|
| Forward 3d realized vol by regime | Welch's t, Mann-Whitney U, Levene, permutation (10 000 shuffles) | 45.3% vs 27.5% (n 169 / 309); Welch, MWU, Levene and permutation p all < 0.0001 |
| Forward 3d return by regime | Welch's t, Mann-Whitney U | −0.61% vs +0.05%; Welch p = 0.081, MWU p = 0.171 |
| Persistence inside each regime | lag-1 autocorrelation over **contiguous same-regime pairs**, z = ρ√n | calm −0.034 (p = 0.55, 298 pairs); elevated +0.023 (p = 0.77, 156 pairs) |
| Behaviour around DVOL spikes (\|z\| > 2) | event study: AR = r − mean(r) over the 60 days ending 4 days before the event; CAR over −3 … +5 | 24 events; CAR(+5) = −5.3%; t ≈ −2.0 to −2.1 on days −3 and −1, ≈ 0 after the event |
| Volatility clustering | GARCH(1,1), Student-t, `arch` | α = 0.107, β = 0.845, persistence 0.952 |
| Two volatility regimes | two-state Markov switching on % returns, R `MSwM`, `set.seed(42)` | low-vol +0.131%/day, sd 1.55%, stay 0.941; high-vol −0.275%/day, sd 3.59%, stay 0.873 |
| Volume and \|return\| | OLS of \|r\| on volume (volume–volatility relation) and the Amihud ratio mean(\|r\| / dollar volume) | OLS p = 2.7e-86, R² = 0.41; Amihud 0.026 per $1B |
| Signed flow → next-day return | OLS (predictive Kyle λ) | p = 0.76, R² = 0.000 |
| Big-move odds vs volume anomaly | in-sample logistic on (volume z, signed-volume z) | P(\|move\| ≥ 2%) 0.315 at z = 0 → 0.390 at z = +3; base 0.316 |
| Out-of-sample directional skill | walk-forward described below | gap = 3: Brier 0.1869 vs base 0.1915; gap = 0: 0.1927 |

Persistence uses contiguous pairs so that a calm day is never paired with
the elevated day that preceded it. Event windows are allowed to overlap (24
events in 481 labelled days), which inflates the t-statistics; they are
reported as descriptive.

## Walk-forward validation (what the code actually does)

`stats_spine.walk_forward_splits(n, n_folds, horizon, gap)`:

1. Rows with a valid target, valid features and a valid regime label are kept
   in time order (477 rows for `fwd_ret_3`).
2. The sample is cut into `n_folds + 1 = 6` equal blocks. Fold *k* tests on
   block *k* (k = 1 … 5) and trains on **every row before it, minus the last
   `gap` rows** — an expanding window.
3. With `gap = horizon = 3` no training label's forward window reaches into
   the test block: `max(train) + horizon < min(test)`. `gap = 0` is the naive
   split in which the last three training labels are computed from prices
   inside the test block.
4. A `StandardScaler` + **uncalibrated** `LogisticRegression` is fitted on the
   training slice only; its out-of-sample probabilities are concatenated over
   the five folds (395 rows) and scored with the Brier loss against a
   base-rate predictor on the same rows.

The horizon is parsed from the target name (`fwd_ret_3 → 3`); passing a
conflicting `horizon` raises `ValueError`. The suite proves the no-overlap
property for n ∈ {100, 477, 731} × horizon ∈ {1, 3, 7}.

This is *not* López de Prado's combinatorial purged CV with an embargo after
the test set; it is an expanding-window walk-forward with a purge-length gap
before each test block. On this sample the gap changes the Brier from 0.1927
to 0.1869 — not the verdict. The test suite includes a synthetic case (20-day
labels, 5-NN on the time index) where removing the gap manufactures skill and
the gap removes it.

## Calibration

`logistic_probability()` fits `CalibratedClassifierCV(sigmoid, TimeSeriesSplit(3))`
around the same pipeline on a single 70/30 time-ordered split and prints a
five-bin reliability curve (Brier 0.2158 on 144 test rows, base rate 0.268).
It is a diagnostic of what calibration does to the probabilities; the
headline skill number is the uncalibrated walk-forward above, and the two are
not comparable (different test sets).

## Limitations that shape the verdict

- **DVOL coverage:** regime results describe 2025-03-20 → 2026-07-13.
- **DVOL is a proxy** for the gamma regime; the GEX-conditioned test needs
  the snapshot history the collector is accumulating.
- **Overlapping windows** (event study, forward returns) make in-sample
  p-values optimistic.
- **The threshold model and the Kyle regressions are in-sample.**
- **Naive dealer-sign convention** (long calls / short puts).

## The verdict in one sentence

On this sample the thesis's **volatility / magnitude** predictions hold
(regime, GARCH, Markov, volume all agree) and its **directional**
predictions do not (regime bias p = 0.08, calm-regime mean reversion p = 0.55,
Kyle λ p = 0.76, walk-forward Brier edge +0.005).
