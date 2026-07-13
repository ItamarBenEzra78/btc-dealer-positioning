"""
gex_engine.py — Compute dealer Gamma Exposure (GEX) from raw Deribit options.

This reproduces the InsiderFinance / SpotGamma methodology from scratch, using
only Deribit's FREE public API. Nothing is scraped from a paid dashboard — we
pull the raw option chain (open interest + implied vol per strike) and compute
every headline metric ourselves:

    Net GEX      Σ call gamma·OI − Σ put gamma·OI   (dealer regime: +damp / −amplify)
    Call Wall    strike with the largest call gamma concentration (resistance/magnet)
    Put Wall     strike with the largest put gamma concentration (support)
    Zero Gamma   spot level where cumulative GEX flips sign (the regime pivot)

GEX per strike ≈ gamma × OI × spot² × 0.01   (dollar delta change per 1% move)
"""

import re
import json
import math
import urllib.request
from datetime import datetime, timezone

import numpy as np

DERIBIT = "https://www.deribit.com/api/v2/public"
# Fixed "now" is injected so the module is deterministic/testable (no hidden clock).
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "gex-engine/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())["result"]


def parse_instrument(name):
    """'BTC-28AUG26-46000-P' -> (expiry_datetime_utc, strike, 'P'|'C')."""
    _, exp, strike, kind = name.split("-")
    day = int(exp[:-5]); mon = MONTHS[exp[-5:-2]]; yr = 2000 + int(exp[-2:])
    expiry = datetime(yr, mon, day, 8, 0, tzinfo=timezone.utc)  # Deribit expires 08:00 UTC
    return expiry, float(strike), kind


def bs_gamma(S, K, T, sigma):
    """Black-Scholes gamma (r=0, standard for crypto). Same gamma for calls & puts."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return pdf / (S * sigma * math.sqrt(T))


def compute_gex(currency="BTC", now_ts=None):
    chain = _get(f"{DERIBIT}/get_book_summary_by_currency?currency={currency}&kind=option")
    # 'now' comes from the freshest quote's timestamp in the feed — no local clock needed.
    now = now_ts or max(c["creation_timestamp"] for c in chain) / 1000.0
    spot = np.median([c["underlying_price"] for c in chain if c.get("underlying_price")])

    per_strike = {}   # strike -> {'call': gex, 'put': gex}
    for c in chain:
        oi, iv, uprice = c.get("open_interest"), c.get("mark_iv"), c.get("underlying_price")
        if not oi or not iv or not uprice:
            continue
        expiry, strike, kind = parse_instrument(c["instrument_name"])
        T = (expiry.timestamp() - now) / (365.25 * 24 * 3600)
        gamma = bs_gamma(uprice, strike, T, iv / 100.0)
        gex = gamma * oi * uprice ** 2 * 0.01
        slot = per_strike.setdefault(strike, {"call": 0.0, "put": 0.0})
        slot["call" if kind == "C" else "put"] += gex

    strikes = sorted(per_strike)
    call_g = np.array([per_strike[k]["call"] for k in strikes])
    put_g = np.array([per_strike[k]["put"] for k in strikes])
    # Dealer convention: long gamma from calls, short gamma from puts (naive SpotGamma model)
    net_by_strike = call_g - put_g

    call_wall = strikes[int(np.argmax(call_g))]
    put_wall = strikes[int(np.argmax(put_g))]

    # Zero gamma: strike where cumulative net GEX crosses zero (regime pivot)
    cum = np.cumsum(net_by_strike)
    zero_gamma = None
    for i in range(1, len(cum)):
        if cum[i - 1] <= 0 < cum[i] or cum[i - 1] >= 0 > cum[i]:
            zero_gamma = strikes[i]
            break

    return {
        "currency": currency,
        "spot": round(float(spot), 2),
        "net_gex": call_g.sum() - put_g.sum(),
        "call_gex": call_g.sum(),
        "put_gex": -put_g.sum(),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "zero_gamma": zero_gamma,
        "regime": "positive (vol dampening)" if (call_g.sum() - put_g.sum()) > 0
                  else "negative (vol amplifying)",
        "n_strikes": len(strikes),
        "asof": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
    }


def _b(x):
    return f"${x/1e9:+.2f}B"


if __name__ == "__main__":
    g = compute_gex("BTC")
    print("=" * 56)
    print(f"  BTC DEALER GAMMA EXPOSURE  (our engine, Deribit data)")
    print("=" * 56)
    print(f"  As of        : {g['asof']}")
    print(f"  Spot         : ${g['spot']:,.0f}")
    print(f"  Net GEX      : {_b(g['net_gex'])}   -> {g['regime']}")
    print(f"  Call GEX     : {_b(g['call_gex'])}")
    print(f"  Put GEX      : {_b(g['put_gex'])}")
    print(f"  Call Wall    : ${g['call_wall']:,.0f}   (resistance / magnet)")
    print(f"  Put Wall     : ${g['put_wall']:,.0f}   (support)")
    zg = f"${g['zero_gamma']:,.0f}" if g['zero_gamma'] else "n/a"
    print(f"  Zero Gamma   : {zg}   (regime pivot)")
    print(f"  Strikes used : {g['n_strikes']}")
