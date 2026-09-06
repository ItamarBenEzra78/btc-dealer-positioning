# Deployment

Three paths, from lightest to heaviest.

## 1. Streamlit Community Cloud — read-only public demo

**Suitability:** the entire dashboard (`dashboard.py`) is safe to publish on
Streamlit Community Cloud. All data sources are free public HTTP endpoints
that do not require an API key: Deribit, CryptoGamma, Forex Factory. The
committed sample data (`data/market_daily.csv`, `data/r_regime.csv`) means the
**Statistical Evidence** tab renders without needing R at runtime.

**What Cloud will run:** only `dashboard.py`. The FastAPI backend (`api.py`)
and the collector worker (`worker.py`) are separate services intended for the
VPS path (see below). The dashboard does not depend on either of them.

**Cloud-safety checklist:**

- ✅ No API keys or secrets — nothing to configure in Streamlit's Secrets UI.
- ✅ No local file writes from the dashboard code path — reads only.
- ✅ Committed sample data — the Evidence and Validation tabs work
  immediately after clone.
- ✅ Deribit and CryptoGamma calls are outbound HTTP with retries and
  graceful fallbacks (see `validate.fetch_oracle`).
- ⚠️ R is *not* available on Streamlit Cloud. `data/r_regime.csv` is
  committed, so the dashboard's Markov section renders anyway; do not try to
  regenerate it on Cloud.
- ⚠️ `arch` (Python GARCH) is only imported lazily inside
  `stats_spine.garch_persistence`, which the dashboard does *not* call — so
  a `pip` failure on that package would not break the dashboard. If it does
  install cleanly (it usually does on Streamlit Cloud), the API layer keeps
  full functionality.

### Pre-flight (verified 2026-09-06)

| Check | Result |
|---|---|
| Entry point | `dashboard.py` at repo root |
| Dependencies | `requirements.txt` at repo root; installs cleanly into a fresh virtualenv |
| Secrets / API keys | **none required** — Deribit, CryptoGamma, Forex Factory are public endpoints |
| Local paths | none; all reads are repo-relative (`data/market_daily.csv`, `data/r_regime.csv`) |
| Writes at runtime | none from the dashboard code path |
| Committed data | `data/market_daily.csv` (731 rows), `data/r_regime.csv` (731 rows) |
| Theme | `.streamlit/config.toml` (dark) — picked up automatically |
| R | not needed at runtime |
| Tests | `python -m pytest -q` → 12 passed; GitHub Actions green on 3.10/3.11/3.12 |

### Click-by-click

1. Open <https://share.streamlit.io> → **Continue with GitHub** → authorise
   Streamlit to read your repositories (one-time OAuth).
2. Top-right **Create app** → **Deploy a public app from GitHub**.
3. Fill in exactly:
   - **Repository:** `ItamarBenEzra78/btc-dealer-positioning`
   - **Branch:** `master`
   - **Main file path:** `dashboard.py`
   - **App URL** (optional): choose a short slug, e.g.
     `btc-dealer-positioning` → gives `https://btc-dealer-positioning.streamlit.app`
4. **Advanced settings** → Python version **3.11** (3.10–3.12 are all CI-tested).
   Leave **Secrets** empty.
5. **Deploy**. First build takes 2–5 minutes (streamlit, plotly, scikit-learn,
   arch). Watch the log; if `arch` fails to build, the dashboard still runs —
   it is only imported lazily by the API layer.
6. When the app shows the **Positioning** tab with live numbers, copy the URL.

### After it is live — exact edits

**README.md** — insert as the first line under the title (before the tests
badge), replacing `<APP-URL>`:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<APP-URL>.streamlit.app)
```

**CV** (`Itamar_Ben_Ezra_Data_Analyst_CV_Final.html`) — the page is at its
height limit, so add the link on the project **title line**, not as a new
line. Change

```html
<span class="t">Statistical Validation of Options-Market Positioning Signals (BTC)</span> <span class="r">Jul 2026</span>
```

to

```html
<span class="t">Statistical Validation of Options-Market Positioning Signals (BTC)</span> <span class="r">Live demo: <a href="https://<APP-URL>.streamlit.app"><APP-URL>.streamlit.app</a> · Jul 2026</span>
```

and in the `.md`, append ` · Live demo: <APP-URL>.streamlit.app` to the same
title line. Re-render the PDF and confirm it is still one page.

Do **not** add the badge or the CV line before the app is live — a broken
link looks worse than no link.

## 2. Local development

See the **Run locally** section in [`README.md`](../README.md). Nothing else
to add.

## 3. Ubuntu VPS with the collector worker — production

The full production setup (worker collecting a snapshot every 30 min,
FastAPI backend, Streamlit dashboard, all managed by `systemd`) is documented
in [`../deploy/DEPLOY.md`](../deploy/DEPLOY.md) and driven by
[`../deploy/setup.sh`](../deploy/setup.sh). One-shot install on a fresh
Ubuntu 22.04/24.04 box; cost is roughly €4–6/month at a small VPS provider.

Use this path if you want the historical GEX snapshot database to keep
growing 24/7 (which is what enables the GEX-native backtests on the
roadmap). Streamlit Community Cloud is only appropriate for the read-only
demo — it does not run background schedulers.

## Environment variables

None are required. Two are optional (see `.env.example`):

| Name | Where used | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | `db.py` | `sqlite:///data/app.db` | SQLAlchemy URL. Point to Postgres for shared/hosted storage without changing any other code. |
| `SNAPSHOT_INTERVAL_MIN` | `worker.py` | `30` | How often the background worker captures a positioning snapshot. |

Neither of these should be committed; use `.env.example` as the template and
never commit `.env`.
