"""
api.py — The backend layer (FastAPI).

Turns the engines from "scripts Streamlit calls" into a real API with typed
contracts. Each engine becomes an endpoint; each response is a Pydantic model
(the abstraction step: loose dicts -> typed domain objects).

Run:   uvicorn api:app --reload --port 8000
Docs:  http://localhost:8000/docs      (auto-generated OpenAPI)
"""

import asyncio
import time
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from positioning import snapshot
from anomalies import pin_setup, add_forward_returns, add_regime_labels
from flow_engine import live_option_flow, add_flow_features, kyle_lambda, significant_threshold
from data_layer import fetch_ohlc
from calendar_layer import next_event, catalyst_read
from validate import fetch_oracle, within
from stats_spine import regime_two_sample, return_persistence, purged_walk_forward, garch_persistence
from phenomenon import load as load_market, analyze_phenomenon
from db import init_db, get_snapshots, get_price_bars

init_db()  # ensure tables exist

app = FastAPI(title="BTC Dealer Positioning API",
              description="Live dealer positioning + statistical evaluation, as a service.",
              version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------
# tiny TTL cache — don't hammer Deribit on every request
# --------------------------------------------------------------------------
_CACHE: Dict[str, tuple] = {}


def cached(key, ttl, fn):
    hit = _CACHE.get(key)
    now = time.time()
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


# --------------------------------------------------------------------------
# Domain models (typed contracts)
# --------------------------------------------------------------------------
class Zones(BaseModel):
    A1: Optional[float]; A2: Optional[float]
    P1: Optional[float]; P2: Optional[float]
    N1: Optional[float]; N2: Optional[float]


class Positioning(BaseModel):
    currency: str
    spot: float
    net_gex: float
    call_gex: float
    put_gex: float
    regime: str
    call_wall: float
    put_wall: float
    zero_gamma: Optional[float]
    max_pain_agg: Optional[float]
    max_pain_near: Optional[float]
    near_expiry: Optional[str]
    near_dte: Optional[float]
    zones: Zones
    n_strikes: int


class PinState(BaseModel):
    conditions: Dict[str, bool]
    conditions_met: int
    pin_likely: bool
    distance_to_max_pain_pct: Optional[float]
    catalyst: Optional[dict]


class Flow(BaseModel):
    n_trades: int
    net_call_premium: float
    net_put_premium: float
    net_premium_flow: float
    flow_tilt: str


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "BTC Dealer Positioning API", "version": "1.0.0",
            "endpoints": ["/positioning/{asset}", "/pin/{asset}", "/flow/{asset}",
                          "/price/{asset}", "/catalyst", "/validate/{asset}",
                          "/stats", "/phenomena"], "docs": "/docs"}


@app.get("/positioning/{asset}", response_model=Positioning)
def get_positioning(asset: str):
    try:
        return cached(f"pos:{asset}", 120, lambda: snapshot(asset.upper()))
    except Exception as e:
        raise HTTPException(502, f"engine error: {e}")


@app.get("/pin/{asset}", response_model=PinState)
def get_pin(asset: str):
    snap = cached(f"pos:{asset}", 120, lambda: snapshot(asset.upper()))
    pin = pin_setup(snap)
    ev = cached("catalyst", 900, next_event)
    pin["catalyst"] = catalyst_read(pin["pin_likely"], ev)
    return pin


@app.get("/flow/{asset}", response_model=Flow)
def get_flow(asset: str):
    return cached(f"flow:{asset}", 120, lambda: live_option_flow(asset.upper()))


@app.get("/overview/{asset}")
async def get_overview(asset: str):
    """Positioning + flow + pin + catalyst, fetched CONCURRENTLY (async).

    The three engine calls each block on Deribit; running them in parallel threads
    makes the aggregate ~as fast as the slowest single call, not their sum.
    """
    a = asset.upper()
    pos, flow, ev = await asyncio.gather(
        asyncio.to_thread(lambda: cached(f"pos:{a}", 120, lambda: snapshot(a))),
        asyncio.to_thread(lambda: cached(f"flow:{a}", 120, lambda: live_option_flow(a))),
        asyncio.to_thread(lambda: cached("catalyst", 900, next_event)),
    )
    pin = pin_setup(pos)
    pin["catalyst"] = catalyst_read(pin["pin_likely"], ev)
    return {"positioning": pos, "flow": flow, "pin": pin}


@app.get("/price/{asset}")
def get_price(asset: str, resolution: str = "60", bars: int = 300):
    df = cached(f"ohlc:{resolution}:{bars}", 120, lambda: fetch_ohlc(resolution, bars))
    return {"asset": asset.upper(), "resolution": resolution,
            "candles": df.assign(date=df["date"].astype(str)).to_dict("records")}


@app.get("/catalyst")
def get_catalyst():
    ev = cached("catalyst", 900, next_event)
    if ev:
        ev = {**ev, "when": ev["when"].isoformat()}
    return {"next_event": ev}


@app.get("/validate/{asset}")
def get_validate(asset: str):
    snap = cached(f"pos:{asset}", 120, lambda: snapshot(asset.upper()))
    try:
        orc = fetch_oracle()
    except Exception as e:
        raise HTTPException(502, f"oracle unreachable: {e}")
    sq, met = orc.get("squeezeLevels", {}), orc.get("metrics", {})
    return {
        "call_wall": {"ours": snap["call_wall"], "oracle": sq.get("resistance"),
                      "match": within(snap["call_wall"], sq.get("resistance"), 0.02)},
        "put_wall": {"ours": snap["put_wall"], "oracle": sq.get("support"),
                     "match": within(snap["put_wall"], sq.get("support"), 0.02)},
        "net_gamma_sign": {"ours": "positive" if snap["net_gex"] > 0 else "negative",
                           "oracle": "positive" if met.get("netGamma", 0) > 0 else "negative"},
    }


def _market():
    def build():
        df = get_price_bars()                       # DB is now the source of truth
        if df.empty:
            df = pd.read_csv("data/market_daily.csv", parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if "logret" not in df.columns:
            df["logret"] = np.log(df["close"]).diff()
        return add_flow_features(add_regime_labels(add_forward_returns(df)))
    return cached("market", 3600, build)


@app.get("/history/{asset}")
def get_history(asset: str, limit: int = 100):
    """Positioning snapshots the collector has persisted (the growing GEX history)."""
    snaps = get_snapshots(asset.upper(), limit)
    return {"asset": asset.upper(), "count": len(snaps), "snapshots": snaps}


@app.get("/stats")
def get_stats():
    df = _market()
    rv = regime_two_sample(df, "fwd_rv_3")
    rp = return_persistence(df)
    pw = purged_walk_forward(df, "fwd_ret_3", ["elevated", "dvol_chg_z", "dvol"])
    try:
        garch = garch_persistence(df)
    except Exception:
        garch = None
    return {
        "n_days": len(df),
        "vol_clustering": {"elevated": rv["mean_elevated"], "calm": rv["mean_calm"],
                           "welch_p": rv["welch_p"], "significant": rv["welch_p"] < 0.05},
        "garch": garch,
        "persistence_calm_autocorr": rp["calm"]["autocorr_lag1"],
        "purged_cv": pw,
        "verdict": "magnitude predictable; direction edge negligible",
    }


@app.get("/phenomena")
def get_phenomena():
    df = _market()
    presets = {
        "dvol_spike": df["dvol_chg_z"] > 2,
        "big_down_day": df["logret"] < -0.03,
        "high_volume": df["vol_z"] > 2,
    }
    return {name: analyze_phenomenon(df, mask, name, verbose=False)
            for name, mask in presets.items()}
