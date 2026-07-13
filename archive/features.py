"""
features.py — The economic glue.

This is the ONLY hand-written domain logic in the project; every other layer
(detection, regime, statistics) is a battle-tested third-party library. Here we
turn raw supply/price numbers into the quantities your thesis is about:

  net_flow        = daily change in total stablecoin supply
                    (USD minted = fresh fiat entering crypto; burned = leaving)
  flow_intensity  = net_flow / stablecoin_supply
                    your core idea: a flow measured RELATIVE to all the money
                    already parked in the system, not in absolute dollars
  fwd_return_k    = BTC forward log-return over the next k days
                    (the effect we later test each anomaly against)
"""

import numpy as np
import pandas as pd


def build_features(csv_path="data/dataset.csv", fwd_horizons=(1, 3, 7)):
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    # --- money in / out of the system -----------------------------------
    df["net_flow"] = df["stablecoin_supply"].diff()

    # --- flow relative to the whole pool (your "ביחס לכלל הכסף") ----------
    df["flow_intensity"] = df["net_flow"] / df["stablecoin_supply"].shift(1)

    # --- BTC log-returns: the target the flow supposedly explains --------
    df["btc_logret"] = np.log(df["btc_price"]).diff()
    for k in fwd_horizons:
        # forward return over the NEXT k days (shifted so no look-ahead leak)
        df[f"fwd_return_{k}"] = (
            np.log(df["btc_price"].shift(-k)) - np.log(df["btc_price"])
        )

    return df


if __name__ == "__main__":
    f = build_features()
    cols = ["date", "stablecoin_supply", "net_flow", "flow_intensity", "fear_greed", "fwd_return_1"]
    print(f[cols].tail(8).to_string(index=False))
    print(f"\n{len(f)} rows, {f.shape[1]} feature columns")
