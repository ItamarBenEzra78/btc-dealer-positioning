"""
data_layer.py — Historical + forward data for the statistical evaluation.

Three jobs:
  1. fetch_price()   BTC daily OHLC (deep, free) -> returns + realized vol
  2. fetch_dvol()    Deribit DVOL implied-vol index history -> vol regime
  3. save_snapshot() append today's live positioning to data/snapshots.jsonl
                     — this is what grows our historical GEX database going forward
                     (deep historical GEX is not free, so we start collecting now).

Run:  python3 data_layer.py           # builds data/market_daily.csv + logs a snapshot
"""

import json
import os
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gex_engine import _get, DERIBIT
from positioning import snapshot

DATA_DIR = "data"
SNAP_FILE = f"{DATA_DIR}/snapshots.jsonl"


def fetch_price(days=730):
    """BTC-PERPETUAL daily OHLC from Deribit (free, deep)."""
    end = _get(f"{DERIBIT}/get_time")            # server time in ms (avoids local clock)
    start = end - days * 86400 * 1000
    url = (f"{DERIBIT}/get_tradingview_chart_data?instrument_name=BTC-PERPETUAL"
           f"&start_timestamp={start}&end_timestamp={end}&resolution=1D")
    r = _get(url)
    df = pd.DataFrame({
        "date": pd.to_datetime(r["ticks"], unit="ms").normalize(),
        "close": r["close"], "high": r["high"], "low": r["low"], "open": r["open"],
        "volume": r["volume"],
    })
    df = df.groupby("date", as_index=False).last().sort_values("date").reset_index(drop=True)
    return df


def fetch_ohlc(resolution="60", bars=300):
    """Intraday/daily BTC-PERPETUAL candles from Deribit for the price chart.

    resolution is a Deribit code: "1","5","10","60" (minutes) or "1D".
    Returns OHLC with a real timestamp (not normalized) for candlestick plotting.
    """
    mins = {"1": 1, "3": 3, "5": 5, "10": 10, "15": 15, "30": 30,
            "60": 60, "120": 120, "1D": 1440}.get(resolution, 60)
    end = _get(f"{DERIBIT}/get_time")
    start = end - bars * mins * 60 * 1000
    r = _get(f"{DERIBIT}/get_tradingview_chart_data?instrument_name=BTC-PERPETUAL"
             f"&start_timestamp={start}&end_timestamp={end}&resolution={resolution}")
    return pd.DataFrame({
        "date": pd.to_datetime(r["ticks"], unit="ms"),
        "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
    })


def fetch_dvol(currency="BTC", days=730):
    """Deribit DVOL implied-volatility index, daily close."""
    end = _get(f"{DERIBIT}/get_time")
    start = end - days * 86400 * 1000
    url = (f"{DERIBIT}/get_volatility_index_data?currency={currency}"
           f"&start_timestamp={start}&end_timestamp={end}&resolution=43200")
    r = _get(url)["data"]
    df = pd.DataFrame(r, columns=["ts", "open", "high", "low", "dvol"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.normalize()
    return df.groupby("date", as_index=False)["dvol"].last()


def build_market_daily(days=730):
    """Combined daily frame: price, log-return, realized vol, DVOL."""
    price = fetch_price(days)
    dvol = fetch_dvol("BTC", days)
    df = price.merge(dvol, on="date", how="left").sort_values("date").reset_index(drop=True)

    df["logret"] = np.log(df["close"]).diff()
    # 7-day annualised realized volatility (%), to compare against DVOL
    df["rv_7d"] = df["logret"].rolling(7).std() * np.sqrt(365) * 100
    df["dvol"] = df["dvol"].ffill()
    return df


def save_snapshot(currency="BTC"):
    """Append today's live positioning snapshot — grows the historical GEX DB."""
    os.makedirs(DATA_DIR, exist_ok=True)
    s = snapshot(currency)
    s["captured_at"] = datetime.fromtimestamp(s["asof"], tz=timezone.utc).isoformat()
    with open(SNAP_FILE, "a") as f:
        f.write(json.dumps(s, default=str) + "\n")
    return s


def load_snapshots():
    if not os.path.exists(SNAP_FILE):
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(SNAP_FILE) if l.strip()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building market_daily...")
    md = build_market_daily(730)
    md.to_csv(f"{DATA_DIR}/market_daily.csv", index=False)
    print(f"  {len(md)} days  ({md['date'].min().date()} -> {md['date'].max().date()})")
    print(md[["date", "close", "rv_7d", "dvol"]].tail(4).to_string(index=False))

    print("\nLogging today's positioning snapshot...")
    s = save_snapshot("BTC")
    n = len(load_snapshots())
    print(f"  saved. snapshot DB now holds {n} day(s).")
    print(f"  spot ${s['spot']:,.0f} · regime {s['regime']} · max_pain_near ${s['max_pain_near']:,.0f}")
