"""
calendar_layer.py — Macro event context for the pin model.

Free economic calendar (Forex Factory JSON via faireconomy). We surface the next
high-impact USD event and encode the friend's insight:

  A max-pain PIN is a low-volatility phenomenon. A scheduled high-impact event
  (CPI, FOMC, NFP) injects implied vol / uncertainty. So if we're pinned AND a
  catalyst is imminent, the pin is a coiled spring -> the odds shift from
  "pinning" toward a DIRECTIONAL BREAK once the number drops.

This is a live context overlay, not a validated signal (event counts are tiny).
"""

import json
import urllib.request
from datetime import datetime, timezone

FEEDS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]


def fetch_calendar(impact="High", countries=("USD",)):
    """Return upcoming high-impact events (UTC), soonest first."""
    events = []
    for url in FEEDS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                for e in json.loads(r.read().decode()):
                    if e.get("impact") != impact:
                        continue
                    if countries and e.get("country") not in countries:
                        continue
                    try:
                        dt = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
                    except Exception:
                        continue
                    events.append({"title": e.get("title"), "country": e.get("country"),
                                   "when": dt, "forecast": e.get("forecast"),
                                   "previous": e.get("previous")})
        except Exception:
            continue
    return sorted(events, key=lambda x: x["when"])


def next_event(now=None, **kw):
    """The soonest upcoming high-impact event + hours until it."""
    now = now or datetime.now(timezone.utc)
    for e in fetch_calendar(**kw):
        if e["when"] > now:
            e = dict(e)
            e["hours_until"] = round((e["when"] - now).total_seconds() / 3600, 1)
            return e
    return None


def catalyst_read(pin_likely, ev, window_h=36):
    """Turn 'pin state + next event' into the friend's directional-break read."""
    if ev is None:
        return {"has_catalyst": False, "message": "No high-impact USD event on the horizon."}
    imminent = ev["hours_until"] <= window_h
    if pin_likely and imminent:
        msg = (f"⚡ {ev['title']} in {ev['hours_until']}h — with a PIN active, this is a "
               f"coiled spring: odds tilt toward a DIRECTIONAL BREAK, not a pin.")
        level = "break-risk"
    elif imminent:
        msg = (f"⚡ {ev['title']} in {ev['hours_until']}h — elevated implied vol reduces pin odds.")
        level = "vol-up"
    else:
        msg = f"Next high-impact: {ev['title']} in {ev['hours_until']}h (far — low effect for now)."
        level = "quiet"
    return {"has_catalyst": True, "imminent": imminent, "level": level,
            "event": ev["title"], "hours_until": ev["hours_until"], "message": msg}


if __name__ == "__main__":
    ev = next_event()
    if ev:
        print(f"Next high-impact USD event: {ev['title']}  in {ev['hours_until']}h  "
              f"(cons {ev['forecast']}, prev {ev['previous']})  @ {ev['when'].isoformat()}")
    for pin in (True, False):
        print(f"\npin_likely={pin}:")
        print("  " + catalyst_read(pin, ev)["message"])
