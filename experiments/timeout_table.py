"""m1 — the 23 (instance, heuristic) pairs excluded by the runtime cap.

Two distinct thresholds operate in this dataset and are easily confused:
  * 600,000 ms  — the feasibility cap. Runs exceeding it are excluded from
                  Stage 1's best-heuristic labeling. 23 pairs exceed it.
  * 1,000,000 ms — the timeout sentinel written when a run did not finish.
                  Rows carrying it have no defined optimality gap. 16 of the
                  23 carry it exactly; the other 7 have genuine measured
                  runtimes between the cap and the sentinel.

Output: results/timeout_instances.csv

Run from repo root: python -m experiments.timeout_table
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_dataset import RUNTIME_CAP_MS, TIMEOUT_SENTINEL_MS

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "Dataset.csv"
RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    over_cap = df[df["runtime_ms"] > RUNTIME_CAP_MS].copy()
    over_cap["hit_sentinel"] = over_cap["runtime_ms"] == TIMEOUT_SENTINEL_MS
    over_cap = over_cap[["file", "n", "type", "heuristic", "runtime_ms", "hit_sentinel"]]
    over_cap = over_cap.sort_values(["n", "heuristic"]).reset_index(drop=True)

    out_path = RESULTS_DIR / "timeout_instances.csv"
    over_cap.to_csv(out_path, index=False)

    print("=" * 78)
    print(f"Runs exceeding the {RUNTIME_CAP_MS:,}ms feasibility cap")
    print("=" * 78)
    print(over_cap.to_string(index=False))
    print()
    print(f"total pairs over cap:            {len(over_cap)}")
    print(f"  of which hit {TIMEOUT_SENTINEL_MS:,}ms sentinel: {int(over_cap['hit_sentinel'].sum())}")
    print(f"  with genuine measured runtime:      {int((~over_cap['hit_sentinel']).sum())}")
    print()
    print("by heuristic:")
    print(over_cap["heuristic"].value_counts().to_string())
    print()
    print(f"n range: {over_cap['n'].min()} to {over_cap['n'].max()}")
    print(f"distinct instances affected: {over_cap['file'].nunique()}")
    print()
    print("by instance (which heuristics timed out):")
    grouped = (
        over_cap.groupby(["file", "n"])
        .agg(heuristics=("heuristic", lambda s: " + ".join(sorted(s))), count=("heuristic", "size"))
        .reset_index()
        .sort_values("n")
    )
    print(grouped.to_string(index=False))
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
