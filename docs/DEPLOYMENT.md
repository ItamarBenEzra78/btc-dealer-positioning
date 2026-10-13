# Deployment

Three paths, from lightest to heaviest.

## 1. Streamlit Community Cloud — read-only public demo

**Suitability:** the dashboard (`dashboard.py`) is safe to publish on
Streamlit Community Cloud. All data sources are free public HTTP endpoints
that do not require an API key (Deribit, Forex Factory). The committed data
(`data/market_daily.csv`, `data/r_regime.csv`, `data/r_regime_summary.json`,
`data/snapshots_sample.jsonl`) means every tab renders without R, without a
database, and — with a visible warning — without network access.

**What Cloud will run:** only `dashboard.py`. The FastAPI backend (`api.py`)
and the collector worker (`worker.py`) are separate services intended for the
VPS path. The dashboard does not depend on either of them.

**Cloud-safety checklist:**

- No API keys or secrets — nothing to configure in Streamlit's Secrets UI.
- No local file writes from the dashboard code path — reads only.
- Committed data — the Evidence tab works immediately after clone.
- Live Deribit calls are wrapped: on failure the Positioning tab shows the
  last stored snapshot (`data/snapshots_sample.jsonl`) and says so.
- The **Validation** tab depends on the CryptoGamma public endpoint, which
  has returned HTTP 401 since 2026-09. The tab shows "Oracle unavailable"
  until that changes; nothing else is affected.
- R is *not* available on Streamlit Cloud. `data/r_regime.csv` and
  `data/r_regime_summary.json` are committed, so the Markov section renders
  anyway; do not try to regenerate them on Cloud.
- `arch` (Python GARCH) is only imported lazily inside
  `stats_spine.garch_persistence`, which the dashboard does not call.

### Pre-flight

| Check | Result |
|---|---|
| Entry point | `dashboard.py` at repo root |
| Dependencies | `requirements.txt` at repo root, fully pinned |
| Secrets / API keys | none required |
| Local paths | none; all reads are repo-relative under `data/` |
| Writes at runtime | none from the dashboard code path |
| Committed data | `market_daily.csv` (731 rows), `r_regime.csv` (731 rows), `r_regime_summary.json`, `snapshots_sample.jsonl` (5 rows) |
| Theme | `.streamlit/config.toml` (dark) — picked up automatically |
| R | not needed at runtime |
| Tests | `python -m pytest -q` — includes an offline `AppTest` render of the dashboard; CI runs ruff + pytest on Python 3.9–3.12 |

### Click-by-click

1. Open <https://share.streamlit.io>, continue with GitHub, authorise
   Streamlit to read your repositories (one-time OAuth).
2. **Create app** → **Deploy a public app from GitHub**.
3. Fill in: repository `ItamarBenEzra78/btc-dealer-positioning`, branch
   `master`, main file path `dashboard.py`, and an optional short app URL.
4. **Advanced settings** → Python version 3.11 (3.9–3.12 are CI-tested).
   Leave **Secrets** empty.
5. **Deploy**. The first build takes a few minutes (streamlit, plotly,
   scikit-learn, arch).
6. When the app shows the **Positioning** tab with numbers, copy the URL and
   add the badge to the top of `README.md`:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<APP-URL>.streamlit.app)
```

Do not add the badge before the app is live.

## 2. Local development

See **Reproduce** in [`README.md`](../README.md).

## 3. Ubuntu VPS with the collector worker — production

The full setup (worker collecting a snapshot every 30 min, FastAPI backend,
Streamlit dashboard, all managed by `systemd`) is documented in
[`../deploy/DEPLOY.md`](../deploy/DEPLOY.md) and driven by
[`../deploy/setup.sh`](../deploy/setup.sh). One-shot install on a fresh
Ubuntu 22.04/24.04 box.

Use this path if you want the historical GEX snapshot database to keep
growing (which is what enables the GEX-native backtests). Streamlit Community
Cloud is only appropriate for the read-only demo — it does not run background
schedulers.

## Environment variables

None are required. Two are optional and read straight from the process
environment (there is no `.env` loader):

| Name | Where used | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | `db.py` | `sqlite:///data/app.db` | SQLAlchemy URL. Point to Postgres (`pip install "psycopg[binary]"`) for shared storage without changing any other code. |
| `SNAPSHOT_INTERVAL_MIN` | `worker.py` | `30` | How often the background worker captures a positioning snapshot. |

Set them in the `systemd` unit (`Environment=...`) or export them in the
shell before starting a service.
