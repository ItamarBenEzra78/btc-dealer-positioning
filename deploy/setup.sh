#!/usr/bin/env bash
#
# setup.sh — one-shot bootstrap for a fresh Ubuntu 22.04/24.04 VPS.
# Run as root:   bash setup.sh
#
# Installs Python + R, creates an app user, clones the repo, builds a venv,
# initialises the database, and starts worker/api/dashboard under systemd.

set -euo pipefail

APP_USER=gex
APP_DIR=/home/$APP_USER/btc-dealer-positioning
REPO=https://github.com/ItamarBenEzra78/btc-dealer-positioning.git

echo ">> [1/8] system packages"
apt-get update -y
apt-get install -y python3 python3-venv python3-pip r-base git ufw

echo ">> [2/8] app user"
id -u "$APP_USER" &>/dev/null || useradd -m -s /bin/bash "$APP_USER"

echo ">> [3/8] clone / update repo"
if [ -d "$APP_DIR/.git" ]; then
  sudo -u "$APP_USER" git -C "$APP_DIR" pull
else
  sudo -u "$APP_USER" git clone "$REPO" "$APP_DIR"
fi

echo ">> [4/8] python venv + deps"
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo ">> [5/8] R packages"
Rscript -e 'install.packages(c("MSwM","changepoint"), repos="https://cloud.r-project.org")' || \
  echo "   (R packages optional — stats still run without them)"

echo ">> [6/8] build initial data + database"
sudo -u "$APP_USER" bash -lc "cd $APP_DIR && .venv/bin/python data_layer.py && .venv/bin/python db.py"

echo ">> [7/8] systemd services"
cp "$APP_DIR"/deploy/*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now worker.service api.service dashboard.service

echo ">> [8/8] firewall"
ufw allow OpenSSH
ufw allow 8000/tcp
ufw allow 8501/tcp
ufw --force enable

IP=$(hostname -I | awk '{print $1}')
echo
echo "=================================================================="
echo "  DONE. Services running under systemd."
echo "  Dashboard : http://$IP:8501"
echo "  API docs  : http://$IP:8000/docs"
echo "  Logs      : journalctl -u worker -f"
echo "=================================================================="
