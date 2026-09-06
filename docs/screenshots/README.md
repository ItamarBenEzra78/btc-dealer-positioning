# Dashboard screenshots — shot list

The main `README.md` references four screenshots and one optional GIF. This
file tells you exactly what to capture, how to crop, and what to name each
file so the `README.md` links resolve.

## Preparation

1. Launch the app locally: `streamlit run dashboard.py`.
2. Wait for the first Deribit fetch (spinner disappears, metrics populate).
3. Use a browser window at **1440 × 900** for consistent output.
4. Zoom to **100%**; disable browser scrollbars if your tool captures them.
5. Recommended tools: **ScreenToGif** (Windows), **Kap** (macOS),
   **Flameshot** (Linux).

## Screenshot 1 — Positioning tab

- **Tab:** `🎯 Positioning` (the default).
- **Filter / state:** Timeframe radio set to `1M`.
- **Crop:** from the four top metrics (Spot, Net GEX, Zero Gamma, Max Pain
  near) through the price-and-dealer-levels candlestick chart. Skip the
  strike-gamma-profile chart below unless it fits without whitespace.
- **Filename:** `docs/screenshots/01-positioning.png`.
- **Caption in README:** *Live spot, Net GEX, Zero-Gamma level, walls, and
  Max Pain with the price chart overlaid.*

## Screenshot 2 — Flow & Threshold tab

- **Tab:** `💵 Flow & Threshold`.
- **Filter / state:** default.
- **Crop:** from the "Live options net-premium flow" subheader through the
  purple **P(|move| ≥ 2% next day)** verdict box.
- **Filename:** `docs/screenshots/02-flow.png`.
- **Caption in README:** *Live options flow, Kyle λ / Amihud impact, CUSUM
  events, and the significant-threshold model.*

## Screenshot 3 — Statistical Evidence tab

- **Tab:** `🔬 Statistical Evidence`.
- **Filter / state:** Markov range set to `1Y`.
- **Crop:** from the "Vol clustering / Directional bias / Persistence" row
  through the **purged walk-forward verdict** box at the bottom. The Markov
  regime chart should be visible in the middle.
- **Filename:** `docs/screenshots/03-evidence.png`.
- **Caption in README:** *The thesis tested on 2 years of BTC — regime tests,
  event study, Markov regime probability, and the leakage-safe verdict.*

## Screenshot 4 — Validation tab

- **Tab:** `✓ Validation`.
- **Filter / state:** default (wait until the CryptoGamma table populates).
- **Crop:** the entire tab — the comparison table plus the interpretation
  box.
- **Filename:** `docs/screenshots/04-validation.png`.
- **Caption in README:** *Reconstructed levels compared side-by-side with
  the CryptoGamma oracle; sign divergences flagged rather than hidden.*

## Optional GIF — 15-second tour

- **Filename:** `docs/screenshots/00-tour.gif`.
- **Length:** aim for 12–15 seconds; keep the file under 4 MB.
- **Sequence:**
  1. Open the app on the **Positioning** tab.
  2. Change the timeframe radio from `1D` to `1M` — the chart re-renders
     with dealer-level dashed lines.
  3. Click the **Statistical Evidence** tab.
  4. Scroll down to reveal the purged-walk-forward verdict block.

That is enough to show a recruiter the flow of the analysis without them
having to launch the app themselves.

## After capturing

- Optimise the PNGs with `oxipng` or `tinypng.com` (target ≤ 400 KB each).
- Optimise the GIF with `gifsicle -O3` (target ≤ 4 MB).
- Do **not** commit the raw `.mp4` recording — GIF only for the README.

## Not committed by default

`.gitignore` excludes `docs/screenshots/*.png` and `docs/screenshots/*.gif`
until you explicitly `git add` them, so an unfinished capture will not sneak
into a commit. Once you are happy with the files, add and commit them
individually.
