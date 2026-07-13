# Research Dossier — Tooling Landscape

Synthesis of 5 parallel research sweeps (2026-07-13). Reliability: 🟢 solid · 🟡 caveats · 🔴 weak/reference.

---

## 1. GEX / Dealer Positioning
**Finding:** Mature GEX logic exists but is equities/CBOE-based. Crypto/Deribit OSS is near-zero. Our Deribit engine already leads.

- **Borrow:** [gex-tracker](https://github.com/Matteo-Ferrara/gex-tracker) 🟢 (GEX math), [gflows](https://github.com/aaguiar10/gflows) 🟢 (vanna/charm), [py_vollib_vectorized](https://pypi.org/project/py-vollib-vectorized/) 🟢 (Black-76 greeks).
- **Data:** [Deribit](https://docs.deribit.com/) 🟢 free · [Tardis.dev](https://tardis.dev/) 🟢 free history · [Laevitas](https://docs.laevitas.ch/options/analytic) 🟡 validation.
- **Gaps:** production crypto GEX (we lead), higher-order VEX/Vanna/Charm, coin-settled greeks, taker-flow dealer-sign model, historical GEX backtest DB.

## 2. Flow & Significant-Threshold
**Finding:** No crypto-native flow/UOA engine in OSS. Price-impact math is mature.

- **Borrow:** [mlfinlab](https://www.mlfinlab.com/en/latest/feature_engineering/micro_features_other.html) 🟢 (Kyle λ, VPIN, CUSUM), [ruptures](https://github.com/deepcharles/ruptures) 🟢 (changepoint), BOCD 🟡 (streaming).
- **Data:** Deribit ticks 🟢, [Tardis](https://tardis.dev/) 🟢, [CoinGlass CVD](https://docs.coinglass.com/) 🟡, [CryptoQuant netflow](https://cryptoquant.com/) 🟡 paid.
- **Gaps:** crypto net-premium-flow engine, **the significant-threshold model (flow→probability) — core IP**, streaming VPIN/OFI, labeled event dataset.

## 3. Macro / Correlation / News
- **Borrow:** [statsmodels](https://www.statsmodels.org/) 🟢 (Granger, coint, Markov), [arch](https://github.com/bashtage/arch) 🟢 (GARCH), [FinBERT/CryptoBERT](https://www.mdpi.com/2504-2289/8/6/63) 🟢.
- **Data:** [FRED](https://fred.stlouisfed.org/docs/api/fred/) 🟢, yfinance 🟡 (+stooq fallback), [GDELT](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) 🟢, [CryptoPanic](https://cryptopanic.com/developers/api/) 🟢, [F&G](https://alternative.me/crypto/api/) 🟢, [DeFiLlama](https://api-docs.defillama.com/) 🟢, [Binance funding](https://developers.binance.com/) 🟢.
- **Note:** true DXY is NOT on FRED (use yfinance `DX-Y.NYB`). BTC-SPX correlation is regime-dependent.
- **Gaps:** dynamic/regime correlation (DCC-GARCH/Markov — no maintained Py lib), unified risk-on/off label, news→signal pipeline.

## 4. Technical Analysis / Levels
- **Borrow:** [TA-Lib](https://github.com/TA-Lib/ta-lib-python) 🟢, [smart-money-concepts](https://github.com/joshyattridge/smart-money-concepts) 🟢 (BOS/CHoCH, 1.9k★), [ruptures](https://github.com/deepcharles/ruptures) 🟢, [trendln](https://github.com/GregoryMorse/trendln) 🟡.
- **Note:** VPOC/HVN = volume analog of option walls. Avoid original `pandas-ta` (archival ~2026).
- **Gaps:** **the confluence engine (TA × options levels merged, ranked, scored) — defining piece, greenfield**, level-object schema, cross-source clustering, regime→level-behavior map.

## 5. Probability / Ensemble / Calibration / Backtest
- **Borrow:** [LightGBM](https://github.com/microsoft/LightGBM) 🟢 (+LogisticRegression 🟢), [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/calibration.html) 🟢, [skfolio CPCV](https://skfolio.org/user_guide/model_selection.html) 🟢, [vectorbt](https://github.com/polakowo/vectorbt) 🟢, [SHAP](https://github.com/shap/shap) 🟢.
- **Kaggle lesson:** LightGBM + feature engineering ≫ model choice + embargoed CV + regime sub-models (G-Research Crypto).
- **Gaps:** calibrated-probability ↔ faithful-SHAP coupling (core IP), calibration under crypto non-stationarity, triple-barrier labeling, regime-conditional ensembling.

---

**Bottom line:** ~80% of plumbing is free & reliable. The moat is the ~20% nobody built for crypto: crypto GEX+higher-order, the significant-threshold model, the confluence engine, dynamic correlation, and calibrated probability with honest factor breakdown.
