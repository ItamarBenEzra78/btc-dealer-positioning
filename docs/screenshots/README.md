# Dashboard screenshots

Three PNGs are committed and referenced from the main `README.md`. They were
captured on 2026-09-06 from the app running locally against live Deribit
data, before the regime-labelling fix; the layout is unchanged but the
numbers in the Evidence screenshot predate the corrected statistics. Recapture
after any change to the Evidence tab.

## How to recapture

1. `streamlit run dashboard.py` and wait for the first Deribit fetch.
2. Browser window at 1440 × 900, zoom 100%.
3. Capture the three views below and overwrite the files in place; the
   README links are by filename.

| File | Tab | State | Crop |
|---|---|---|---|
| `01-positioning.png` | Positioning | timeframe `1M` | from the four top metrics through the price-and-levels chart |
| `02-flow.png` | Flow & Threshold | default | from the net-premium-flow subheader through the `P(\|move\| ≥ 2%)` verdict box |
| `03-evidence.png` | Statistical Evidence | Markov range `1Y` | from the three regime tiles through the walk-forward verdict box |

## Validation tab

No screenshot is committed. The CryptoGamma public endpoint has returned
HTTP 401 since 2026-09, so the tab shows "Oracle unavailable". If the endpoint
comes back, capture the whole tab as `04-validation.png` and add a row to the
README table.

## Housekeeping

- Keep each PNG at or under ~400 KB (`oxipng` or an online optimiser).
- PNGs under `docs/screenshots/` are tracked; only `*.mp4` recordings are
  git-ignored.
