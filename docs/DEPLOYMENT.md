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

### Manual steps to deploy

1. In your GitHub repo, confirm that `dashboard.py`, `requirements.txt`,
   `.streamlit/config.toml`, `data/market_daily.csv`, and `data/r_regime.csv`
   are all committed.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **New app** → pick this repo → branch `master` → main file
   `dashboard.py` → **Deploy**.
4. First boot takes 2–5 minutes while dependencies install. Watch the log; if
   `arch` fails to build, that is fine — it is not needed by the dashboard.
5. Once the app is up, copy the public URL. Add it to the top of `README.md`
   as an active Streamlit demo badge.

### After deploying — add the demo badge

Only *after* the URL is live, add this line under the title in `README.md`
(replace the placeholder URL):

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-URL.streamlit.app)
```

Do **not** add the badge before the app is live — a broken badge looks worse
than no badge.

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
