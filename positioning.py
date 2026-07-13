"""
positioning.py — Full dealer-positioning snapshot from Deribit (L1).

Extends the base GEX engine with the two metrics the source article emphasised
most, plus the group's A/P/N vocabulary:

  Max Pain      per-expiry strike that minimises total option intrinsic value
                (where the most options expire worthless — the "pin" magnet).
  A1 / A2       Absolute-gamma zones: strikes with the largest total gamma.
  P1 / P2       Positive-gamma zones (dealers dampen — magnets/mean-reversion).
  N1 / N2       Negative-gamma zones (dealers amplify — accelerants).

Everything is computed from the free Deribit public API. Dealer sign is the
naive convention (long calls / short puts) — flagged, not hidden.
"""

import numpy as np

from gex_engine import _get, parse_instrument, bs_gamma, DERIBIT


def build_chain(currency="BTC", now_ts=None):
    """Return per-option records with greeks, plus spot and the 'as-of' timestamp."""
    raw = _get(f"{DERIBIT}/get_book_summary_by_currency?currency={currency}&kind=option")
    now = now_ts or max(c["creation_timestamp"] for c in raw) / 1000.0
    spot = float(np.median([c["underlying_price"] for c in raw if c.get("underlying_price")]))

    records = []
    for c in raw:
        oi, iv, uprice = c.get("open_interest"), c.get("mark_iv"), c.get("underlying_price")
        if not oi or not iv or not uprice:
            continue
        expiry, strike, kind = parse_instrument(c["instrument_name"])
        T = (expiry.timestamp() - now) / (365.25 * 24 * 3600)
        if T <= 0:
            continue
        gamma = bs_gamma(uprice, strike, T, iv / 100.0)
        records.append({
            "expiry": expiry, "strike": strike, "kind": kind,
            "oi": oi, "iv": iv, "T": T,
            "gex": gamma * oi * uprice ** 2 * 0.01,   # $ notional per 1% move
        })
    return records, spot, now


def max_pain(records, expiry=None):
    """Strike minimising total intrinsic value paid to holders (the pin level).

    If `expiry` is given, restrict to that expiry (the article stresses per-expiry,
    usually the monthly). Otherwise aggregate the whole chain.
    """
    recs = [r for r in records if expiry is None or r["expiry"] == expiry]
    if not recs:
        return None
    strikes = sorted({r["strike"] for r in recs})
    best_k, best_pain = None, None
    for K in strikes:
        pain = 0.0
        for r in recs:
            if r["kind"] == "C":
                pain += r["oi"] * max(0.0, K - r["strike"])   # call holders' payout
            else:
                pain += r["oi"] * max(0.0, r["strike"] - K)   # put holders' payout
        if best_pain is None or pain < best_pain:
            best_pain, best_k = pain, K
    return best_k


def gamma_by_strike(records):
    """Aggregate call/put/net GEX per strike."""
    per = {}
    for r in records:
        slot = per.setdefault(r["strike"], {"call": 0.0, "put": 0.0})
        slot["call" if r["kind"] == "C" else "put"] += r["gex"]
    strikes = sorted(per)
    call = np.array([per[k]["call"] for k in strikes])
    put = np.array([per[k]["put"] for k in strikes])
    return strikes, call, put, call - put   # net = call - put (dealer convention)


def apn_zones(strikes, call, put, net):
    """A = top absolute-gamma strikes; P = top positive-net; N = most negative-net."""
    total_abs = call + put
    order_abs = np.argsort(total_abs)[::-1]
    order_pos = np.argsort(net)[::-1]
    order_neg = np.argsort(net)
    pick = lambda order, i: (float(strikes[order[i]]) if i < len(order) else None)
    return {
        "A1": pick(order_abs, 0), "A2": pick(order_abs, 1),
        "P1": pick(order_pos, 0), "P2": pick(order_pos, 1),
        "N1": pick(order_neg, 0), "N2": pick(order_neg, 1),
    }


def zero_gamma_flip(records, spot, lo=0.75, hi=1.25, steps=140):
    """Proper gamma flip: the SPOT level where aggregate dealer gamma exposure = 0.

    We recompute every option's gamma at a grid of hypothetical spot prices (holding
    time & IV fixed), sum the dealer-signed exposure, and find the zero crossing
    nearest the current spot. This is the correct 'zero gamma', not a strike-space
    cumulative sum.
    """
    grid = np.linspace(spot * lo, spot * hi, steps)
    net = np.empty(steps)
    for j, S in enumerate(grid):
        g = 0.0
        for r in records:
            sign = 1.0 if r["kind"] == "C" else -1.0        # naive dealer convention
            g += sign * bs_gamma(S, r["strike"], r["T"], r["iv"] / 100.0) * r["oi"] * S ** 2 * 0.01
        net[j] = g
    crossings = []
    for i in range(1, steps):
        if net[i - 1] == 0 or (net[i - 1] < 0) != (net[i] < 0):
            x0, x1, y0, y1 = grid[i - 1], grid[i], net[i - 1], net[i]
            x = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else grid[i]
            crossings.append(float(x))
    if not crossings:
        return None
    return min(crossings, key=lambda x: abs(x - spot))       # flip nearest spot


def snapshot(currency="BTC", now_ts=None):
    records, spot, now = build_chain(currency, now_ts)
    strikes, call, put, net = gamma_by_strike(records)

    # nearest expiry with meaningful OI (for the per-expiry max pain / pin thesis)
    expiries = sorted({r["expiry"] for r in records})
    near_expiry = expiries[0] if expiries else None

    net_gex = float(call.sum() - put.sum())
    return {
        "currency": currency,
        "asof": now,
        "spot": round(spot, 2),
        "net_gex": net_gex,
        "call_gex": float(call.sum()),
        "put_gex": float(-put.sum()),
        "regime": "positive" if net_gex > 0 else "negative",
        "call_wall": float(strikes[int(np.argmax(call))]),
        "put_wall": float(strikes[int(np.argmax(put))]),
        "zero_gamma": zero_gamma_flip(records, spot),
        "max_pain_agg": max_pain(records),
        "max_pain_near": max_pain(records, near_expiry),
        "near_expiry": near_expiry.date().isoformat() if near_expiry else None,
        "near_dte": round((near_expiry.timestamp() - now) / 86400.0, 1) if near_expiry else None,
        "zones": apn_zones(strikes, call, put, net),
        "n_strikes": len(strikes),
    }


if __name__ == "__main__":
    from datetime import datetime, timezone
    s = snapshot("BTC")
    z = s["zones"]
    print("=" * 58)
    print(f"  BTC DEALER POSITIONING  ·  spot ${s['spot']:,.0f}")
    print("=" * 58)
    print(f"  Net GEX     {s['net_gex']/1e9:+.3f}B   regime: {s['regime']}")
    print(f"  Zero Gamma  ${s['zero_gamma']:,.0f}" if s['zero_gamma'] else "  Zero Gamma  n/a")
    print(f"  Call Wall   ${s['call_wall']:,.0f}      Put Wall   ${s['put_wall']:,.0f}")
    print(f"  Max Pain    agg ${s['max_pain_agg']:,.0f}   near({s['near_expiry']}, {s['near_dte']}d) ${s['max_pain_near']:,.0f}")
    print(f"  A-zones     A1 ${z['A1']:,.0f}   A2 ${z['A2']:,.0f}")
    print(f"  P-zones     P1 ${z['P1']:,.0f}   P2 ${z['P2']:,.0f}")
    print(f"  N-zones     N1 ${z['N1']:,.0f}   N2 ${z['N2']:,.0f}")
    print(f"  strikes {s['n_strikes']}  ·  as-of {datetime.fromtimestamp(s['asof'], tz=timezone.utc).isoformat()}")
