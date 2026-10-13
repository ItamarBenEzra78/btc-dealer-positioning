# Case study — testing a popular market thesis

*A one-page narrative for readers who want the story rather than the code.*

## The problem

Retail options dashboards draw confident-looking levels on a chart: a *Call
Wall* here, a *Put Wall* there, a *Zero-Gamma* pivot, a *Max Pain* magnet.
The implicit promise is that these levels predict where price will go. I could
not find a published statistical test of that promise, so I built one: **if I
rebuild these metrics from raw data and test them properly, what survives?**

## The approach

1. **Rebuild, don't scrape.** Every headline metric is recomputed from the free
   Deribit option chain (open interest and implied vol per strike) using
   Black-Scholes gamma.
2. **Turn each claim into a testable statement.** "Positive-gamma tapes are
   calmer" becomes *forward realized volatility differs between regimes*.
   "Calm tapes mean-revert" becomes *lag-1 autocorrelation is negative inside
   the calm regime*. "Big flow moves price" becomes *P(|move| ≥ 2%) rises on
   high-volume days*. "Positioning predicts direction" becomes *a
   walk-forward classifier beats the base rate*.
3. **Keep the test set clean.** Forward-return labels overlap in time, so the
   walk-forward leaves a horizon-length gap before each test block.
4. **Validate against an outsider.** Reconstructed levels were compared to a
   free third-party oracle (CryptoGamma) while it was reachable.

## What I found

Numbers are from `stats_spine.py`, `flow_engine.py` and `inference.R` on the
frozen sample (`data/sample_2024-07_2026-07.csv`). Regime tests cover the 481
days with a DVOL label (2025-03-20 → 2026-07-13).

| Claim | Verdict |
|---|---|
| Vol regimes differ in forward realized vol | **Yes** — 45.3% vs 27.5%, Welch p < 0.0001 |
| Volatility clustering is persistent | **Yes** — GARCH(1,1) persistence 0.952 |
| Two distinct regimes exist | **Yes** — Markov switching, stay 0.94 / 0.87 |
| Calm tapes mean-revert | **No** — autocorr −0.034, p = 0.55 (see below) |
| Regime carries a 3-day directional bias | **No** — Welch p = 0.081 |
| High volume raises odds of a big move | **In-sample yes** — 0.315 → 0.390 |
| Flow predicts tomorrow's direction | **No** — Kyle λ p = 0.76 |
| Positioning predicts **direction** out of sample | **No** — Brier 0.1869 vs base 0.1915 |
| Rebuilt levels match an external oracle | **Matched on 2026-07-13** (call wall $65k = $65k); endpoint 401 since 2026-09 |

## The result that changed

An earlier version of this project reported that calm tapes mean-revert
(autocorr −0.091, p = 0.03) and presented it as weak support for the thesis.
That result came from a labelling bug: days with no DVOL value (the first 231
rows of the sample, plus the warm-up of the rolling median) were being
labelled *calm* because a NaN comparison evaluates to False. With those days
excluded, the calm sample shrinks from 559 to 309 days, vol clustering gets
*stronger* (calm forward vol falls from 35% to 28%), and the mean-reversion
effect disappears. The correction is recorded here rather than absorbed
silently, because the discipline of reporting a result that vanished is the
same discipline that makes the surviving results worth anything.

## What the gap did and did not do

The README once claimed that "purged CV changed the verdict". The numbers do
not support that: with a 3-day gap before each test block the model's Brier
is 0.1869; without the gap it is 0.1927; the base rate is 0.1915. For a
three-feature linear model on this sample the gap is good hygiene, not a
plot twist. The claim is now stated as what it is — the verdict (no
directional skill) holds with or without it — and the test suite contains a
synthetic case where the gap *does* matter, so the mechanism is demonstrated
rather than asserted.

## What I'd tell a hiring manager

- I can take a vague, popular claim and turn it into specific tests.
- I know the difference between *statistically significant* and *useful*.
- I report the results that disappeared when a bug was fixed.
- I build the plumbing (DB, worker, API, tests, CI, deploy) so the analysis
  can keep running after I stop looking at it.

Full technical detail: [`METHODOLOGY.md`](METHODOLOGY.md) · code:
[`../README.md`](../README.md)
