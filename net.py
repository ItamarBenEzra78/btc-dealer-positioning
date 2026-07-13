"""
net.py — Async network layer (httpx + tenacity).

A resilient, concurrent HTTP helper for new fetching code. tenacity gives clean
exponential-backoff retries (replacing hand-rolled try/except loops); httpx +
asyncio let many requests run at once instead of one-at-a-time.
"""

import httpx
from tenacity import (retry, stop_after_attempt, wait_exponential,
                      retry_if_exception_type)

DERIBIT = "https://www.deribit.com/api/v2/public"
_UA = {"User-Agent": "gex-net/1.0"}


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((httpx.HTTPError,)),
       reraise=True)
async def aget_json(url, timeout=20):
    """GET JSON with automatic exponential-backoff retries on network errors."""
    async with httpx.AsyncClient(timeout=timeout, headers=_UA) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def deribit_index(currency="BTC"):
    """Example async Deribit call — the current index price."""
    name = f"{currency.lower()}_usd"
    data = await aget_json(f"{DERIBIT}/get_index_price?index_name={name}")
    return data["result"]["index_price"]


if __name__ == "__main__":
    import asyncio
    px = asyncio.run(deribit_index("BTC"))
    print(f"async httpx OK — BTC index ${px:,.0f}")
