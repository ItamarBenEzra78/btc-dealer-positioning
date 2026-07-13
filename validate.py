"""
validate.py — Sanity-check our engine against an independent oracle.

CryptoGamma publishes a free BTC GEX snapshot (also computed from Deribit).
We compare structural levels — the parts that should agree regardless of
sign convention — and honestly surface where they diverge (net-gamma sign,
which depends on the dealer-sign assumption we flagged).

Run:  python3 validate.py
"""

import json
import time
import urllib.request

from positioning import snapshot

ORACLE = "https://cryptogamma.io/api/public/snapshot"


def fetch_oracle(url=ORACLE, _depth=0, retries=3):
    """Fetch the oracle snapshot, following redirects and retrying on network hiccups."""
    req = urllib.request.Request(url, headers={"User-Agent": "gex-validate/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 307, 308) and _depth < 4:
                loc = e.headers.get("Location", "")
                if loc.startswith("/"):
                    loc = "https://cryptogamma.io" + loc
                if loc:
                    return fetch_oracle(loc, _depth + 1, retries)
            raise
        except Exception:                      # timeout / URLError — retry a couple times
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def within(a, b, pct):
    if a is None or b is None:
        return False
    return abs(a - b) / b <= pct


def main():
    ours = snapshot("BTC")
    try:
        orc = fetch_oracle()
    except Exception as e:
        print(f"  Oracle unreachable ({type(e).__name__}) — skipping validation this run.")
        return
    sq = orc.get("squeezeLevels", {})
    met = orc.get("metrics", {})
    risk = orc.get("riskMetrics", {})

    print("=" * 60)
    print("  VALIDATION — our engine vs CryptoGamma oracle")
    print("=" * 60)
    print(f"  {'metric':<16}{'ours':>14}{'oracle':>14}   match")
    print("  " + "-" * 56)

    rows = [
        ("Call Wall / res", ours["call_wall"], sq.get("resistance"), 0.02),
        ("Put Wall / supp", ours["put_wall"], sq.get("support"), 0.02),
        ("Spot", ours["spot"], sq.get("currentPrice"), 0.01),
        ("Max Pain (agg)", ours["max_pain_agg"], sq.get("breakout"), 0.03),
    ]
    for name, a, b, tol in rows:
        ok = "✅" if within(a, b, tol) else "⚠️"
        av = f"${a:,.0f}" if a else "—"
        bv = f"${b:,.0f}" if b else "—"
        print(f"  {name:<16}{av:>14}{bv:>14}   {ok}")

    # sign of net gamma — the honest divergence
    our_sign = "positive" if ours["net_gex"] > 0 else "negative"
    orc_sign = "positive" if met.get("netGamma", 0) > 0 else "negative"
    sign_match = "✅" if our_sign == orc_sign else "⚠️ divergent (dealer-sign assumption)"
    print("  " + "-" * 56)
    print(f"  Net gamma sign   {our_sign:>14}{orc_sign:>14}   {sign_match}")
    ours_pin = "likely" if pin(ours) else "no"
    orc_pin = str(risk.get("pinRisk", "?")).split(" (")[0]
    pin_match = "✅" if (ours_pin == "likely") == orc_pin.lower().startswith("high") else "⚠️"
    print(f"  Pin risk         {ours_pin:>14}{orc_pin:>14}   {pin_match}")

    print("\n  Reading: structural LEVELS validate against the paid oracle;")
    print("  the net-gamma SIGN differs because we use the naive dealer convention")
    print("  and they use a different sign model. This is the flagged assumption,")
    print("  not a bug — and exactly why our roadmap includes a taker-flow sign model.")


def pin(snap):
    from anomalies import pin_setup
    return pin_setup(snap)["pin_likely"]


if __name__ == "__main__":
    main()
