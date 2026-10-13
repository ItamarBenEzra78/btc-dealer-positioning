"""
run_all.py — Run the whole pipeline end to end, in order.

    python3 run_all.py

Fetches fresh history from Deribit, logs a snapshot, runs the flow model, the
statistical evaluation, the R inference, and the oracle validation.

Files this OVERWRITES:  data/market_daily.csv, data/r_regime.csv,
                        data/r_regime_summary.json, data/app.db
Files this APPENDS to:  data/snapshots/snapshots.jsonl (git-ignored)
Never touched:          data/sample_2024-07_2026-07.csv — the frozen sample
                        the tests and the documented numbers are tied to.

The dashboard reads market_daily.csv, so its figures move after a refresh;
the README figures do not. (`streamlit run dashboard.py` to launch it.)
"""

import subprocess
import sys


def step(title, cmd):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    # trim noisy warnings for a clean run log
    for line in out.splitlines():
        if any(w in line for w in ("Warning", "warn(", "NotOpenSSL", "convergence",
                                   "raw_prediction", "iteration")):
            continue
        print(line)
    if r.returncode != 0:
        print(f"  [!] step exited {r.returncode}")
    return r.returncode


def main():
    ok = 0
    ok += step("1/5  Build history + log a positioning snapshot", [sys.executable, "data_layer.py"])
    ok += step("2/5  Flow + significant-threshold model", [sys.executable, "flow_engine.py"])
    ok += step("3/5  Statistical evaluation (the core)", [sys.executable, "stats_spine.py"])
    ok += step("4/5  Markov regime inference (R)", ["Rscript", "inference.R"])
    ok += step("5/5  Validation vs CryptoGamma oracle", [sys.executable, "validate.py"])

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE" if ok == 0 else f"  PIPELINE FINISHED with {ok} non-zero step(s)")
    print("  Launch the dashboard:  streamlit run dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
