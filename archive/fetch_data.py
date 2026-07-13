"""
fetch_data.py — Collect the raw signals for the capital-flow anomaly model.

Three free, key-less public sources:
  1. CoinGecko      -> BTC daily price, trading volume, market cap
  2. DeFiLlama      -> total stablecoin supply (proxy for fiat entering crypto)
  3. alternative.me -> Crypto Fear & Greed Index (market psychology)

Everything is merged on the calendar date (UTC) and written to data/dataset.csv.
Run:  python3 fetch_data.py
"""

import time
import io
import json
import urllib.request
import pandas as pd

DATA_DIR = "data"


def _get_json(url, retries=3, pause=2.0):
    """Fetch a URL and parse JSON, with a few polite retries for flaky APIs."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "crypto-flow-research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - we want to retry on anything network-ish
            last_err = e
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_btc_market():
    """BTC daily price / volume / market cap for the last 365 days (free-tier max)."""
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        "?vs_currency=usd&days=365&interval=daily"
    )
    raw = _get_json(url)

    def to_series(pairs, name):
        s = pd.DataFrame(pairs, columns=["ts_ms", name])
        s["date"] = pd.to_datetime(s["ts_ms"], unit="ms").dt.normalize()
        return s[["date", name]]

    price = to_series(raw["prices"], "btc_price")
    vol = to_series(raw["total_volumes"], "btc_volume")
    mcap = to_series(raw["market_caps"], "btc_market_cap")

    df = price.merge(vol, on="date").merge(mcap, on="date")
    # daily endpoint sometimes returns two rows for "today"; keep the last
    df = df.groupby("date", as_index=False).last()
    return df


def fetch_stablecoin_supply():
    """Total USD-pegged stablecoin supply per day = money parked inside crypto."""
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    raw = _get_json(url)
    rows = []
    for r in raw:
        rows.append(
            {
                "date": pd.to_datetime(int(r["date"]), unit="s").normalize(),
                "stablecoin_supply": r["totalCirculatingUSD"]["peggedUSD"],
            }
        )
    return pd.DataFrame(rows)


def fetch_fear_greed():
    """Daily Fear & Greed index (0 = extreme fear, 100 = extreme greed), full history."""
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    raw = _get_json(url)
    rows = []
    for r in raw["data"]:
        rows.append(
            {
                "date": pd.to_datetime(int(r["timestamp"]), unit="s").normalize(),
                "fear_greed": int(r["value"]),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("1/3  BTC market data (CoinGecko)...")
    btc = fetch_btc_market()
    print(f"     {len(btc)} days")

    print("2/3  Stablecoin supply (DeFiLlama)...")
    stables = fetch_stablecoin_supply()
    print(f"     {len(stables)} days")

    print("3/3  Fear & Greed index (alternative.me)...")
    fng = fetch_fear_greed()
    print(f"     {len(fng)} days")

    # Inner-join on BTC's 365-day window; that is our analysis period.
    df = btc.merge(stables, on="date", how="left").merge(fng, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # Forward-fill the two slower-moving series so there are no gaps.
    df[["stablecoin_supply", "fear_greed"]] = df[["stablecoin_supply", "fear_greed"]].ffill()

    out = f"{DATA_DIR}/dataset.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows x {df.shape[1]} cols -> {out}")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
