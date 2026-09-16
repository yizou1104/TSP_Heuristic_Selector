"""Regenerate every Stage 1 / Stage 2 number the manuscript quotes, on the
CURRENT codebase (post S2.1 feature renormalization + S2.2 instance-grouped
conformal calibration), so the paper can be updated against verified values
rather than stale pre-fix ones.

Prints, for each of the four pipelines: MAE, RMSE, log-space R^2, empirical
coverage, mean/median interval width, within-tolerance rates, regret stats,
conformal score, calibration instance counts, the misclassified test
instances, the instances falling outside their prediction interval, and the
top gap-regressor feature importances.

Usage:
    python experiments/refresh_paper_numbers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_PATH  # noqa: E402
from src.models import (  # noqa: E402
    _STAGE2_FIT_FUNCTIONS,
    prepare_stage2_split,
    run_integrated_pipeline,
)

PIPELINES = [("rf", "quantile"), ("rf", "conformal"), ("xgb", "quantile"), ("xgb", "conformal")]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def label(b, u):
    return f"{'RF' if b == 'rf' else 'XGB'}+{'Quantile' if u == 'quantile' else 'Conformal'}"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    pd.set_option("display.width", 250)

    split = prepare_stage2_split(df)
    print(f"Split: {len(split['train_files'])} train instances / "
          f"{len(split['test_files'])} test instances (SEED=1)")

    # Calibration instance counts for the two conformal pipelines (S2.2)
    print("\n=== S2.2: instance-grouped calibration split sizes ===")
    for b in ("rf", "xgb"):
        fit = _STAGE2_FIT_FUNCTIONS[(b, "conformal")](split["df_gap"], split["train_files"])
        print(
            f"{label(b, 'conformal')}: {fit['n_train_proper_instances']} train-proper instances, "
            f"{fit['n_cal_instances']} calibration instances, "
            f"conformal_score={fit['conformal_score']:.4f}"
        )

    summary_rows = []
    for b, u in PIPELINES:
        name = label(b, u)
        r = run_integrated_pipeline(df, b, u)
        res = r["results"]

        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        print(f"  classifier_accuracy : {r['classifier_accuracy']:.4f} "
              f"({int(round(r['classifier_accuracy'] * len(res)))}/{len(res)})")
        print(f"  gap_mae             : {r['gap_mae']:.4f} pp")
        print(f"  gap_rmse            : {r['gap_rmse']:.4f} pp")
        print(f"  r2_log              : {r['r2_log']:.4f}")
        print(f"  coverage            : {r['coverage']:.4f} "
              f"({int(round(r['coverage'] * len(res.dropna(subset=['true_gap']))))}"
              f"/{len(res.dropna(subset=['true_gap']))})")
        print(f"  mean_interval_width : {r['mean_interval_width']:.2f} pp")
        print(f"  median_interval_wid : {r['median_interval_width']:.2f} pp")
        print(f"  within_tolerance    : {r['within_tolerance']}")
        print(f"  regret mean/med/max : {r['regret_mean']:.4f} / "
              f"{r['regret_median']:.4f} / {r['regret_max']:.4f}")
        print(f"  catastrophic_rate   : {r['catastrophic_failure_rate']:.4f}")
        print(f"  conformal_score     : {r['conformal_score']}")

        wrong = res[~res["classifier_correct"]]
        print(f"\n  Misclassified test instances ({len(wrong)}):")
        if len(wrong):
            print(wrong[["file", "true_heuristic", "predicted_heuristic", "regret"]]
                  .to_string(index=False))

        outside = res.dropna(subset=["true_gap"])
        outside = outside[~outside["in_interval"]]
        print(f"\n  Test instances OUTSIDE prediction interval ({len(outside)}):")
        if len(outside):
            print(outside[["file", "true_gap", "predicted_gap", "lower_gap", "upper_gap"]]
                  .to_string(index=False))

        print("\n  Top gap-regressor feature importances:")
        print(r["gap_feature_importance"].head(6).to_string(index=False))

        summary_rows.append({
            "pipeline": name,
            "mae_pp": r["gap_mae"],
            "rmse_pp": r["gap_rmse"],
            "r2_log": r["r2_log"],
            "coverage": r["coverage"],
            "median_width_pp": r["median_interval_width"],
            "mean_width_pp": r["mean_interval_width"],
            "classifier_accuracy": r["classifier_accuracy"],
            "conformal_score": r["conformal_score"],
        })

    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("UPDATED STAGE 2 COMPARISON TABLE (for the manuscript)")
    print("=" * 70)
    print(summary.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "paper_numbers_refreshed.csv"
    summary.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
