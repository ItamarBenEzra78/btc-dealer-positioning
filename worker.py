"""
worker.py — The background worker (APScheduler).

Runs the collection jobs on a schedule, unattended. This is where the project's
biggest data gap gets solved: free historical GEX doesn't exist, so we build it
ourselves by capturing a positioning snapshot every few minutes, forever.

Jobs:
    collect_snapshot   every SNAPSHOT_INTERVAL_MIN (default 30) -> DB
    refresh_market     daily 00:30 UTC: rebuild price/DVOL history -> DB

Run:  python3 worker.py
Config:  SNAPSHOT_INTERVAL_MIN=15 python3 worker.py
"""

import functools
import os
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

import db
from data_layer import save_snapshot, build_market_daily

print = functools.partial(print, flush=True)   # worker logs must appear live
SNAPSHOT_MIN = int(os.getenv("SNAPSHOT_INTERVAL_MIN", "30"))


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def collect_snapshot():
    try:
        s = save_snapshot("BTC")            # writes jsonl + DB
        print(f"[{_now()}] snapshot  spot ${s['spot']:,.0f}  {s['regime']}  "
              f"maxpain ${s['max_pain_near']:,.0f}  · DB total {len(db.get_snapshots())}", flush=True)
    except Exception as e:
        print(f"[{_now()}] snapshot FAILED: {e}")


def refresh_market():
    try:
        md = build_market_daily(730)
        db.upsert_price_bars(md)
        print(f"[{_now()}] market refreshed: {len(md)} price bars -> DB")
    except Exception as e:
        print(f"[{_now()}] market refresh FAILED: {e}")


def main():
    db.init_db()
    sched = BlockingScheduler(timezone="UTC")
    # capture immediately on start, then every SNAPSHOT_MIN minutes
    sched.add_job(collect_snapshot, "interval", minutes=SNAPSHOT_MIN,
                  next_run_time=datetime.now(timezone.utc))
    sched.add_job(refresh_market, "cron", hour=0, minute=30)
    print(f"worker started · snapshot every {SNAPSHOT_MIN}min · market refresh daily 00:30 UTC")
    print("(Ctrl+C to stop)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("worker stopped.")


if __name__ == "__main__":
    main()
