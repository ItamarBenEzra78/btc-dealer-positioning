# Deploy on a VPS (Linux + systemd)

Runs the collector 24/7 so the GEX history actually accumulates, plus the API and
dashboard — all managed by systemd (auto-restart on crash/reboot).

## 1. Get a VPS (~$5/month)

Any Ubuntu 22.04/24.04 box works:
- **Hetzner** CX22 (~€4/mo) — best value
- **DigitalOcean** / **Vultr** / **Linode** basic droplet ($5–6/mo)

Pick Ubuntu 24.04, 1 vCPU / 2 GB RAM is plenty. Note the server's IP.

## 2. Connect

```bash
ssh root@YOUR_SERVER_IP
```

## 3. One-shot install

```bash
curl -fsSL https://raw.githubusercontent.com/ItamarBenEzra78/btc-dealer-positioning/master/deploy/setup.sh -o setup.sh
bash setup.sh
```

That installs Python + R, creates the `gex` user, clones the repo, builds a venv,
initialises the database, and starts all three services. Takes ~3–5 minutes.

When it finishes it prints your URLs:
- Dashboard → `http://YOUR_SERVER_IP:8501`
- API docs → `http://YOUR_SERVER_IP:8000/docs`

## 4. Verify it's running

```bash
systemctl status worker api dashboard      # all three should be "active (running)"
journalctl -u worker -f                      # watch the collector capture snapshots live
```

You should see a snapshot line every 30 minutes. That is the GEX history building.

## Everyday commands

```bash
# logs
journalctl -u worker -f            # collector
journalctl -u api -f               # backend
journalctl -u dashboard -f         # UI

# control
systemctl restart worker           # restart a service
systemctl stop dashboard           # stop one

# update to latest code
sudo -u gex git -C /home/gex/btc-dealer-positioning pull
systemctl restart worker api dashboard
```

## Tune the collection interval

Edit `Environment=SNAPSHOT_INTERVAL_MIN=30` in `/etc/systemd/system/worker.service`,
then `systemctl daemon-reload && systemctl restart worker`.

## Optional upgrades (when ready)

- **Domain + HTTPS**: put nginx in front (`apt install nginx`), reverse-proxy
  8501/8000, add a free Let's Encrypt cert with `certbot`. Then close 8000/8501 in ufw.
- **Postgres instead of SQLite**: spin up managed Postgres (Neon/Supabase free tier),
  set `Environment=DATABASE_URL=postgresql+psycopg://...` in `worker.service` and
  `api.service`, `pip install "psycopg[binary]"`, restart. Nothing else changes.
- **Backups**: `cron` a nightly copy of `data/app.db` (or use Postgres backups).

## Security note

Ports 8000/8501 are open to the internet (fine — it's public market data). If you
want the dashboard private, put it behind nginx + basic auth, or restrict the ufw
rules to your IP.
