"""S2.3 ablation: how much do Stage 1's classification errors cost Stage 2's
gap-prediction accuracy?

For each of the four backbone/uncertainty_method pipelines, runs
``run_integrated_pipeline`` twice:
  - "Real":   use_true_heuristic=False — Stage 2 is fed Stage 1's actual
              predicted heuristic (the real end-to-end pipeline).
  - "Oracle": use_true_heuristic=True  — Stage 2 is fed the TRUE best
              heuristic, simulating a perfect Stage 1.

The gap between Real and Oracle isolates how much of Stage 2's error is
attributable to Stage 1's classification mistakes vs. the gap regressor's
own noise.

Usage:
    python experiments/stage1_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_PATH  # noqa: E402
from src.models import run_integrated_pipeline  # noqa: E402

PIPELINES = [
    ("rf", "quantile"),
    ("rf", "conformal"),
    ("xgb", "quantile"),
    ("xgb", "conformal"),
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV_PATH = RESULTS_DIR / "stage1_ablation.csv"

METRIC_KEYS = ["gap_mae", "gap_rmse", "r2_log", "coverage", "classifier_accuracy"]


def _pipeline_label(backbone: str, uncertainty_method: str) -> str:
    backbone_label = "RF" if backbone == "rf" else "XGB"
    method_label = "Quantile" if uncertainty_method == "quantile" else "Conformal"
    return f"{backbone_label}+{method_label}"


def run_ablation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backbone, uncertainty_method in PIPELINES:
        label = _pipeline_label(backbone, uncertainty_method)

        real = run_integrated_pipeline(
            df, backbone, uncertainty_method, use_true_heuristic=False
        )
        oracle = run_integrated_pipeline(
            df, backbone, uncertainty_method, use_true_heuristic=True
        )

        row = {"pipeline": label}
        for key in METRIC_KEYS:
            row[f"real_{key}"] = real[key]
            row[f"oracle_{key}"] = oracle[key]
        row["mae_delta"] = real["gap_mae"] - oracle["gap_mae"]
        rows.append(row)

    return pd.DataFrame(rows)


def print_table(results_df: pd.DataFrame) -> None:
    header = (
        f"{'Pipeline':14s} | {'Real MAE':>9s} | {'Oracle MAE':>10s} | {'MAE Delta':>9s} | "
        f"{'Real R2':>8s} | {'Oracle R2':>9s} | {'Real Coverage':>13s} | {'Oracle Coverage':>15s}"
    )
    print(header)
    print("-" * len(header))
    for _, r in results_df.iterrows():
        print(
            f"{r['pipeline']:14s} | {r['real_gap_mae']:9.2f} | {r['oracle_gap_mae']:10.2f} | "
            f"{r['mae_delta']:9.2f} | {r['real_r2_log']:8.3f} | {r['oracle_r2_log']:9.3f} | "
            f"{r['real_coverage']:13.3f} | {r['oracle_coverage']:15.3f}"
        )
    print()

    for _, r in results_df.iterrows():
        real_mae = r["real_gap_mae"]
        oracle_mae = r["oracle_gap_mae"]
        delta = real_mae - oracle_mae
        relative_pct = (delta / oracle_mae * 100) if oracle_mae != 0 else float("nan")
        print(
            f"{r['pipeline']}: Stage 1 errors cost {delta:.2f}pp of MAE "
            f"({relative_pct:.1f}% relative increase)."
        )


def main() -> None:
    if not Path(DATASET_PATH).exists():
        sys.exit(f"Dataset not found at {DATASET_PATH}. Build it via python -m src.build_dataset first.")

    print(f"Loading {DATASET_PATH} ...")
    df = pd.read_csv(DATASET_PATH)

    results_df = run_ablation(df)
    print_table(results_df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"Saved comparison table to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
