"""
gex_engine.py — Primitives for dealer Gamma Exposure (GEX) from raw Deribit options.

Shared building blocks used by positioning.py (the full snapshot engine):

    _get()              thin JSON GET against Deribit's free public API
    parse_instrument()  'BTC-28AUG26-46000-P' -> (expiry, strike, kind)
    bs_gamma()          Black-Scholes gamma (r = 0)

GEX per strike ≈ gamma × OI × spot² × 0.01   (dollar delta change per 1% move).
The aggregation (walls, zero-gamma, max pain, zones) lives in positioning.py.
"""

import json
import math
import urllib.request
from datetime import datetime, timezone

DERIBIT = "https://www.deribit.com/api/v2/public"
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "gex-engine/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())["result"]


def parse_instrument(name):
    """'BTC-28AUG26-46000-P' -> (expiry_datetime_utc, strike, 'P'|'C')."""
    _, exp, strike, kind = name.split("-")
    day, mon, yr = int(exp[:-5]), MONTHS[exp[-5:-2]], 2000 + int(exp[-2:])
    expiry = datetime(yr, mon, day, 8, 0, tzinfo=timezone.utc)  # Deribit expires 08:00 UTC
    return expiry, float(strike), kind


def bs_gamma(S, K, T, sigma):
    """Black-Scholes gamma (r=0, standard for crypto). Same gamma for calls & puts."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return pdf / (S * sigma * math.sqrt(T))
