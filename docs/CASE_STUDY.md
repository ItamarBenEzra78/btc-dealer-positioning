# Case study — testing a popular market thesis without fooling myself

*A one-page narrative for readers who want the story rather than the code.*

## The problem

Retail options dashboards (SpotGamma-style "Gamma Exposure" tools) draw
confident-looking levels on a chart: a *Call Wall* here, a *Put Wall* there, a
*Zero-Gamma* pivot, a *Max Pain* magnet. The implicit promise is that these
levels predict where price will go. I could not find a published statistical
test of that promise, so I wanted to answer it myself: **if I rebuild these
metrics from raw data and test them properly, what survives?**

## The approach

1. **Rebuild, don't scrape.** Every headline metric is recomputed from the free
   Deribit option chain (open interest and implied vol per strike) using
   Black-Scholes gamma. No paid feed, no black box.
2. **Turn each claim into a testable statement.** "Positive-gamma tapes are
   calmer" becomes *forward realized volatility differs between regimes*.
   "Big flow moves price" becomes *P(|move| ≥ 2%) rises on high-flow days*.
   "Positioning predicts direction" becomes *a leakage-safe classifier beats
   the base rate*.
3. **Refuse to leak.** Forward-return labels overlap in time. A naive split
   lets tomorrow's price into today's training set. I used a purged,
   embargoed walk-forward so that never happens — and it changed the answer.
4. **Validate against an outsider.** Reconstructed levels are compared to a
   free third-party oracle (CryptoGamma) so I would catch my own bugs.

## What I built

- A Python pipeline: ingestion → SQLite/SQLAlchemy store → engines → statistics.
- An R step for the one model R does best (Markov regime-switching, `MSwM`).
- A four-tab Streamlit dashboard and a FastAPI backend with typed contracts.
- A background collector that keeps building the historical GEX dataset that
  does not exist for free anywhere.
- 12 `pytest` tests, a one-shot VPS bootstrap, and `systemd` units.

## What I found

| Claim | Verdict |
|---|---|
| Vol regimes differ in forward realized vol | ✅ strong (Welch p < 0.001) |
| Calm tapes mean-revert | ✅ weak but significant (autocorr −0.091, p = 0.03) |
| Vol spikes cluster with drawdowns | ✅ (CAR ≈ −6.5%) |
| Two distinct regimes exist | ✅ (Markov switching, sticky 0.94) |
| High flow raises odds of a big move | ✅ (0.32 → 0.39) |
| Positioning predicts **direction** | ➖ **no** — Brier 0.187 vs base 0.192 |
| Rebuilt levels match a paid oracle | ✅ (Call Wall $65k = $65k) |

## Why the negative result is the point

The single most useful output is the row that says *no*. A tool that only
showed the six green rows would look better and be worse. The directional
edge — the thing people actually pay for — is within noise once leakage is
removed. Reporting that plainly is what makes the rest of the results
trustworthy.

## What I'd tell a hiring manager

- I can take a vague, popular claim and turn it into specific tests.
- I know the difference between *statistically significant* and *useful*.
- I build the plumbing (DB, worker, API, tests, deploy) so the analysis can
  keep running after I stop looking at it.
- I would rather ship an honest null than a flattering chart.

Full technical detail: [`METHODOLOGY.md`](METHODOLOGY.md) · code:
[`../README.md`](../README.md)
