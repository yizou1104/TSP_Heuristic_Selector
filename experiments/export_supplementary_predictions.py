"""S3.2 — export two supplementary prediction CSVs for the revision.

File 1: results/supplementary_stage1_oof_predictions.csv (160 rows, one per
instance) — out-of-fold Stage 1 predictions from all three selectors
(Logistic Regression, Random Forest, XGBoost), joined side by side. Regret
is left as NaN wherever it is genuinely undefined (no feasible cost recorded
for the predicted heuristic on that instance) — this is what substantiates
the paper's 160/159/159 regret denominators (MI-6).

File 2: results/supplementary_stage2_predictions.csv (32 rows, one per
Seed-1 held-out test instance) — full Stage 2 detail for all four
integrated pipelines (RF/XGB x Quantile/Conformal), side by side.

Does not retrain with different settings, does not change any hyperparameter
or seed. SEED=1 throughout, matching the paper's reference partition.

Run from repo root: python -m experiments.export_supplementary_predictions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.features import STAGE1_INSTANCE_FEATURES
from src.models import (
    SEED,
    build_random_forest_pipeline,
    build_xgboost_tuned_pipeline,
    build_logistic_regression_pipeline,
    evaluate_stage1_cv,
    prepare_stage1_data,
    run_integrated_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "Dataset.csv"
RESULTS_DIR = REPO_ROOT / "results"
PAPER_NUMBERS_PATH = RESULTS_DIR / "paper_numbers_refreshed.csv"

EXPECTED_DENOMINATORS = {"lr": 160, "rf": 159, "xgb": 159}

# (backbone, uncertainty_method) -> label used in paper_numbers_refreshed.csv
PIPELINE_LABELS = {
    ("rf", "quantile"): "RF+Quantile",
    ("rf", "conformal"): "RF+Conformal",
    ("xgb", "quantile"): "XGB+Quantile",
    ("xgb", "conformal"): "XGB+Conformal",
}


def load_df() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


# ---------------------------------------------------------------------------
# File 1 — Stage 1 out-of-fold predictions, 160 rows
# ---------------------------------------------------------------------------
def _lr_oof_predictions(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Reproduces the out-of-fold-prediction portion of evaluate_stage1_cv()
    for Logistic Regression only, WITHOUT calling gini_feature_importance()
    (LogisticRegression has no .feature_importances_, so the shared helper
    crashes on this model). Same StratifiedKFold(n_splits=5, shuffle=True,
    random_state=seed) and regret computation as evaluate_stage1_cv — only
    the importance step is skipped. No modeling logic changed.
    """
    df_prepped, best_rows = prepare_stage1_data(df)
    X = best_rows[STAGE1_INSTANCE_FEATURES]
    y_true = best_rows["heuristic"]

    pipeline = build_logistic_regression_pipeline(seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_oof = cross_val_predict(pipeline, X, y_true, cv=cv)

    pred_cost_lookup = (
        df_prepped[df_prepped["feasible"]][["file", "heuristic", "cost"]]
        .drop_duplicates()
        .set_index(["file", "heuristic"])["cost"]
    )
    best_cost_per_instance = best_rows.set_index("file")["best_feasible_cost"]

    regret_df = pd.DataFrame(
        {
            "file": best_rows["file"].values,
            "true_heuristic": y_true.values,
            "predicted_heuristic": y_oof,
        }
    )
    regret_df["pred_cost"] = regret_df.apply(
        lambda r: pred_cost_lookup.get((r["file"], r["predicted_heuristic"]), np.nan), axis=1
    )
    regret_df["best_cost"] = regret_df["file"].map(best_cost_per_instance)
    regret_df["regret"] = regret_df["pred_cost"] / regret_df["best_cost"] - 1

    return {"oof_predictions": regret_df}


def build_stage1_oof(df: pd.DataFrame) -> pd.DataFrame:
    lr = _lr_oof_predictions(df, seed=SEED)
    rf = evaluate_stage1_cv(df, build_random_forest_pipeline, seed=SEED)
    xgb = evaluate_stage1_cv(
        df, build_xgboost_tuned_pipeline, seed=SEED, needs_label_encoding=True
    )

    def prep(result: dict, prefix: str) -> pd.DataFrame:
        oof = result["oof_predictions"][["file", "true_heuristic", "predicted_heuristic", "pred_cost", "regret"]].copy()
        oof = oof.rename(
            columns={
                "predicted_heuristic": f"{prefix}_predicted",
                "pred_cost": f"{prefix}_pred_cost",
                "regret": f"{prefix}_regret",
            }
        )
        return oof

    lr_oof = prep(lr, "lr")
    rf_oof = prep(rf, "rf").drop(columns=["true_heuristic"])
    xgb_oof = prep(xgb, "xgb").drop(columns=["true_heuristic"])

    merged = lr_oof.merge(rf_oof, on="file", how="outer").merge(xgb_oof, on="file", how="outer")

    # n / type / best_feasible_cost come from best_rows (identical across all three fits)
    best_rows = lr["oof_predictions"][["file"]].copy()
    # Pull n / type / best_feasible_cost from the dataset's best-heuristic rows directly.
    from src.models import prepare_stage1_data

    _, best = prepare_stage1_data(df)
    meta = best[["file", "n", "type", "best_feasible_cost"]].drop_duplicates("file")
    merged = meta.merge(merged, on="file", how="right")

    cols = [
        "file", "n", "type", "true_heuristic", "best_feasible_cost",
        "lr_predicted", "lr_pred_cost", "lr_regret",
        "rf_predicted", "rf_pred_cost", "rf_regret",
        "xgb_predicted", "xgb_pred_cost", "xgb_regret",
    ]
    merged = merged[cols].sort_values("file").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# File 2 — Stage 2 integrated predictions, 32 rows
# ---------------------------------------------------------------------------
def build_stage2_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per Seed-1 test instance, with a COMPLETE column block per
    pipeline.

    Each pipeline gets its own `true_gap` column. This matters: `true_gap` is
    the realised optimality gap of the heuristic THAT pipeline's Stage 1
    selected, so it differs between the RF and XGB backbones on any instance
    where they disagree (3 of the 32 at Seed 1). Sharing a single `true_gap`
    column across pipelines — as an earlier version of this script did —
    silently mismatches the XGB predictions against the RF backbone's targets
    and makes the exported table disagree with the paper's MAE and R^2.
    """
    outs = {}
    for backbone, method in PIPELINE_LABELS:
        outs[(backbone, method)] = run_integrated_pipeline(
            df, backbone=backbone, uncertainty_method=method, seed=SEED, use_true_heuristic=False
        )

    PER_PIPELINE = [
        ("predicted_heuristic", "predicted_heuristic"),
        ("classifier_correct", "classifier_correct"),
        ("true_gap", "true_gap"),
        ("predicted_gap", "predicted_gap"),
        ("lower_gap", "lower"),
        ("upper_gap", "upper"),
        ("interval_width", "interval_width"),
        ("gap_error", "gap_error"),
        ("in_interval", "covered"),
    ]

    merged = None
    ordered_cols = []
    for backbone, method in PIPELINE_LABELS:
        prefix = f"{backbone}_{method}"
        src_cols = ["file"] + [c for c, _ in PER_PIPELINE]
        block = outs[(backbone, method)]["results"][src_cols].rename(
            columns={c: f"{prefix}_{suffix}" for c, suffix in PER_PIPELINE}
        )
        ordered_cols += [f"{prefix}_{suffix}" for _, suffix in PER_PIPELINE]
        merged = block if merged is None else merged.merge(block, on="file", how="outer")

    # `true_heuristic` is the genuinely best feasible heuristic and is
    # identical across pipelines, so it is carried once.
    truth = outs[("rf", "quantile")]["results"][["file", "true_heuristic"]]
    merged = truth.merge(merged, on="file", how="right")

    from src.models import prepare_stage1_data

    _, best = prepare_stage1_data(df)
    meta = best[["file", "n", "type"]].drop_duplicates("file")
    merged = meta.merge(merged, on="file", how="right")

    cols = ["file", "n", "type", "true_heuristic"] + ordered_cols
    merged = merged[cols].sort_values("file").reset_index(drop=True)
    return merged, outs


def consistency_guard(stage2_df: pd.DataFrame, outs: dict) -> pd.DataFrame:
    """Re-derive MAE / R2(log) / coverage from the exported 32-row table and
    compare against results/paper_numbers_refreshed.csv to 4 decimals."""
    from sklearn.metrics import mean_absolute_error, r2_score

    expected = pd.read_csv(PAPER_NUMBERS_PATH).set_index("pipeline")

    rows = []
    ok = True
    for (backbone, method), label in PIPELINE_LABELS.items():
        prefix = f"{backbone}_{method}"
        # each pipeline carries its OWN true_gap (see build_stage2_predictions)
        valid = stage2_df.dropna(subset=[f"{prefix}_true_gap", f"{prefix}_predicted_gap"])
        mae = mean_absolute_error(valid[f"{prefix}_true_gap"], valid[f"{prefix}_predicted_gap"])
        r2log = r2_score(np.log1p(valid[f"{prefix}_true_gap"]), np.log1p(valid[f"{prefix}_predicted_gap"]))
        coverage = valid[f"{prefix}_covered"].mean()

        exp_mae = expected.loc[label, "mae_pp"]
        exp_r2 = expected.loc[label, "r2_log"]
        exp_cov = expected.loc[label, "coverage"]

        mae_match = round(mae, 4) == round(exp_mae, 4)
        r2_match = round(r2log, 4) == round(exp_r2, 4)
        cov_match = round(coverage, 4) == round(exp_cov, 4)
        ok = ok and mae_match and r2_match and cov_match

        rows.append(
            {
                "pipeline": label,
                "mae_exported": round(mae, 4), "mae_expected": round(exp_mae, 4), "mae_match": mae_match,
                "r2log_exported": round(r2log, 4), "r2log_expected": round(exp_r2, 4), "r2log_match": r2_match,
                "coverage_exported": round(coverage, 4), "coverage_expected": round(exp_cov, 4), "coverage_match": cov_match,
            }
        )

    table = pd.DataFrame(rows)
    return table, ok


def main() -> None:
    if not PAPER_NUMBERS_PATH.exists():
        print(f"PRECONDITION FAILED: {PAPER_NUMBERS_PATH} does not exist. Stopping.")
        sys.exit(1)
    expected_check = pd.read_csv(PAPER_NUMBERS_PATH)
    if len(expected_check) != 4:
        print(f"PRECONDITION FAILED: expected 4 rows in {PAPER_NUMBERS_PATH}, found {len(expected_check)}. Stopping.")
        sys.exit(1)

    df = load_df()

    print("=" * 70)
    print("TASK 1a: Stage 1 out-of-fold predictions (File 1)")
    print("=" * 70)
    stage1_df = build_stage1_oof(df)
    out1 = RESULTS_DIR / "supplementary_stage1_oof_predictions.csv"
    stage1_df.to_csv(out1, index=False)
    print(f"wrote {out1}  ({len(stage1_df)} rows)")

    counts = {
        "lr": int(stage1_df["lr_regret"].notna().sum()),
        "rf": int(stage1_df["rf_regret"].notna().sum()),
        "xgb": int(stage1_df["xgb_regret"].notna().sum()),
    }
    print(f"non-null regret counts: {counts}")
    assertion_ok = True
    for key, expected_n in EXPECTED_DENOMINATORS.items():
        if counts[key] != expected_n:
            assertion_ok = False
            print(f"  MISMATCH: {key} expected {expected_n}, got {counts[key]}")
    if not assertion_ok:
        print("ASSERTION FAILED on regret denominators. Stopping before writing further files.")
        sys.exit(1)
    print(f"ASSERTION PASSED: LR=160, RF={counts['rf']}, XGB={counts['xgb']} match paper's stated denominators.")

    print()
    print("=" * 70)
    print("TASK 1b: Stage 2 integrated predictions (File 2)")
    print("=" * 70)
    stage2_df, outs = build_stage2_predictions(df)
    assert len(stage2_df) == 32, f"expected 32 rows, got {len(stage2_df)}"
    out2 = RESULTS_DIR / "supplementary_stage2_predictions.csv"
    stage2_df.to_csv(out2, index=False)
    print(f"wrote {out2}  ({len(stage2_df)} rows)")

    print()
    print("=" * 70)
    print("CONSISTENCY GUARD: exported vs. paper_numbers_refreshed.csv")
    print("=" * 70)
    table, ok = consistency_guard(stage2_df, outs)
    print(table.to_string(index=False))
    if not ok:
        print()
        print("CONSISTENCY GUARD FAILED. Not overwriting/finalizing — investigate before reuse.")
        sys.exit(1)
    print()
    print("CONSISTENCY GUARD PASSED: all four pipelines match to 4 decimal places.")


if __name__ == "__main__":
    main()
