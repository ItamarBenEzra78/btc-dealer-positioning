# BTC Dealer Positioning — Statistical Evidence Engine

A crypto options analytics tool built on one principle: **take the qualitative claims
of dealer-positioning theory (gamma exposure, max pain, gamma regimes) and test them
rigorously against data** — instead of just drawing pretty levels on a chart.

The dashboards that *display* GEX exist (InsiderFinance, CryptoGamma, Glassnode).
Almost none *statistically evaluate* whether the phenomena carry real, measurable signal.
That evaluation — honest, calibrated, backtested, leakage-safe — is the whole point.

> **Thesis × statistics.** The thesis (market-maker hedging creates structure) comes from
> practitioner theory. The value we add is the rigorous test: does it hold up, and by how much?

## Headline finding (the honest verdict)

On 2 years of BTC, dealer-flow and regime signals predict **volatility / magnitude**
(highly significant) but the **directional** edge is negligible under purged walk-forward CV
(Brier 0.186 vs base-rate 0.192 — a ~0.005 improvement, within noise).

That weak result on direction is a feature: most retail tools quietly hide it. Reporting
both is what separates a research instrument from a toy — and it matches market-microstructure
theory (volume → volatility is strong; volume → direction is weak).

## What it does

- Live **dealer positioning** from Deribit: Net GEX, Call/Put Wall, Zero Gamma, **Max Pain**
  (per-expiry), and **A/P/N gamma zones** — plus a live check of the article's **pin conditions**.
- Live **net-premium flow** + a **significant-threshold model**: how much flow is enough to
  raise the odds of a big move (Kyle λ, Amihud, CUSUM).
- A **statistical evaluation** of the regime thesis: event studies, two-sample regime tests,
  return-persistence, **Markov regime-switching (R)**, and a **purged/embargoed** skill check.
- **Validation** against the CryptoGamma oracle.
- A **Streamlit dashboard** unifying all of it.

## Architecture

| Module | Layer | Role |
|--------|-------|------|
| `gex_engine.py` | L1 | Base GEX from Deribit (Black-Scholes gamma, walls, zero-gamma) |
| `positioning.py` | L1 | Full snapshot: **+ Max Pain + A/P/N zones** |
| `flow_engine.py` | L2 | **Net-premium flow + significant-threshold model** (Kyle λ, Amihud, CUSUM) |
| `data_layer.py` | L0 | BTC/DVOL history (2y) + **forward snapshot collector** |
| `anomalies.py` | L2 | Event detection (vol-regime, DVOL spikes) + live pin check |
| `stats_spine.py` | ★ | Event study, regime tests, permutation, calibration, **purged walk-forward CV** |
| `inference.R` | ★ | **Markov regime-switching** (MSwM) — the heavy inference R does best |
| `validate.py` | ✓ | Sanity-check our engine vs the CryptoGamma oracle |
| `dashboard.py` | UI | Streamlit dashboard (4 tabs) |

Python for pipelines and engines, R for inference — a deliberate polyglot split.
Only the engines are hand-written; statistics ride on `scipy`, `statsmodels`, `scikit-learn`, R's `MSwM`.

## Run

```bash
pip install -r requirements.txt
Rscript -e 'install.packages(c("MSwM","changepoint"))'   # R side

python3 data_layer.py     # build 2y history + log a positioning snapshot
python3 flow_engine.py     # flow + significant-threshold model
python3 stats_spine.py     # the statistical evaluation (prints all results)
Rscript inference.R         # Markov regime model -> data/r_regime.csv
python3 validate.py         # compare our levels to the CryptoGamma oracle
streamlit run dashboard.py  # the live dashboard
```

## Findings (2024-07 → 2026-07, BTC)

| Claim tested | Test | Result | Verdict |
|---|---|---|---|
| Regimes have different volatility | Welch + Levene on forward RV | elevated **45%** vs calm **35%**, p=**0.0001** | ✅ strong |
| Positive-gamma/calm tape mean-reverts | lag-1 autocorrelation | calm **−0.091** (p=0.03) | ✅ |
| Vol spikes cluster with drawdowns | event study (CAR) | CAR **−6.5%** around spikes | ✅ |
| Two distinct regimes exist | Markov switching (R) | low-vol +0.13%/day sticky(0.94); high-vol −0.28%/day | ✅ formal |
| Flow raises odds of a **big move** | Amihud + logistic | P(|move|≥2%): 0.32 → **0.39** on high-flow days | ✅ magnitude |
| Flow predicts **direction** | predictive Kyle λ + purged CV | Kyle p=0.76; edge negligible (Brier 0.186 vs 0.192) | ➖ no reliable edge |
| Our levels match a paid oracle | vs CryptoGamma | Call Wall **$65k = $65k**, spot & max pain agree | ✅ validated |

## Honest limitations

- **Dealer sign** uses the naive convention (long calls / short puts) — flagged, and it is exactly
  why our net-gamma *sign* diverges from CryptoGamma while the *levels* match. Roadmap: taker-flow sign model.
- **DVOL ≠ gamma flip.** Regime tests use implied vol as an observable proxy; the true GEX-conditioned
  test needs the historical GEX the forward collector is now building.
- **GARCH (rugarch) skipped** — unavailable as a binary on R 4.6; volatility clustering confirmed instead
  via Levene (p=0.009) and the Markov model.
- **Not financial advice.** A research tool that reports what the data says — including when it says "no signal."

## Roadmap

- [ ] Forward GEX snapshot DB → GEX-native backtests (zero-gamma break, pin rate)
- [ ] Taker-flow dealer-sign model (resolve the sign divergence)
- [ ] Macro layer (BTC-SPX regime correlation) with R `estudy2`
- [ ] Higher-order flows (Vanna/Charm exposure)
```
