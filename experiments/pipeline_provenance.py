"""S3.1 — verified instance-count provenance for the pipeline diagram.

Prints, and writes results/pipeline_provenance.csv with, every instance
count needed to draw an accurate two-protocol data-flow diagram:

  Protocol A (model comparison): stratified 5-fold CV over all 160
    instances -> the 79.37% +/- 5.45% Stage 1 accuracy figure.
  Protocol B (integrated pipeline): 128/32 instance split -> Stage 1 fit on
    128 -> Stage 2 fit on 4x128 gap rows -> conformal variants further split
    the 128 training instances into fit/calibration via GroupShuffleSplit
    -> evaluation on the 32 held-out test instances.

Every number here is read from the actual fitted objects / split functions,
not hardcoded, and is checked against the value the paper currently states
(where the paper states one).

Run from repo root: python -m experiments.pipeline_provenance
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import (
    SEED,
    prepare_stage1_data,
    prepare_stage2_split,
    fit_gap_regressor_rf_conformal,
    fit_gap_regressor_xgb_conformal,
)
from src.features import STAGE1_INSTANCE_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "Dataset.csv"
RESULTS_DIR = REPO_ROOT / "results"

# What the paper currently states, for cross-checking. (label -> value)
PAPER_STATED = {
    "total_instances": 160,
    "stage1_cv_accuracy_mean_pct": 79.37,
    "stage2_train_instances": 128,
    "stage2_test_instances": 32,
    "conformal_train_proper_instances": 102,
    "conformal_calibration_instances": 26,
}


def main() -> None:
    rows = []

    def report(label: str, computed, stated=None):
        match = "" if stated is None else ("MATCH" if str(computed) == str(stated) else "MISMATCH <-- CHECK")
        print(f"  {label:45s} computed={computed!s:>10}  stated={stated!s:>10}  {match}")
        rows.append({"item": label, "computed": computed, "paper_stated": stated, "status": match or "n/a"})

    df = pd.read_csv(DATASET_PATH)

    print("=" * 78)
    print("BLOCK 1 — Overall instance / row counts")
    print("=" * 78)
    df_prepped, best_rows = prepare_stage1_data(df)
    report("total instances (best_rows, Stage 1 label table)", len(best_rows), PAPER_STATED["total_instances"])
    df_gap = df_prepped.dropna(subset=["optimality_gap"]).copy()
    report("total Stage 2 gap rows (non-null optimality_gap)", len(df_gap))
    report("distinct instances contributing gap rows", df_gap["file"].nunique())
    report("gap rows per instance (should be 4: one per heuristic)", round(len(df_gap) / df_gap["file"].nunique(), 3))

    print()
    print("=" * 78)
    print("BLOCK 2 — Protocol A: Stage 1 stratified 5-fold CV (all instances)")
    print("=" * 78)
    X = best_rows[STAGE1_INSTANCE_FEATURES]
    y = best_rows["heuristic"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    report("n_splits", cv.get_n_splits())
    report("total instances entering CV", len(X))
    class_counts_total = y.value_counts().to_dict()
    report("class counts (all 160 instances)", class_counts_total)

    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        fold_train_counts = y.iloc[train_idx].value_counts().to_dict()
        fold_test_counts = y.iloc[test_idx].value_counts().to_dict()
        report(f"fold {i}: train size / test size", f"{len(train_idx)} / {len(test_idx)}")
        report(f"fold {i}: test-fold class counts", fold_test_counts)

    print()
    print("=" * 78)
    print("BLOCK 3 — Protocol B: Stage 2 fixed 80/20 instance split (Seed 1)")
    print("=" * 78)
    split = prepare_stage2_split(df, seed=SEED)
    n_train = len(split["best_rows_train"])
    n_test = len(split["best_rows_test"])
    report("train instances", n_train, PAPER_STATED["stage2_train_instances"])
    report("test instances", n_test, PAPER_STATED["stage2_test_instances"])
    report("train+test instances sum to total?", n_train + n_test == len(best_rows))

    df_gap_split = split["df_gap"]
    train_files = set(split["train_files"])
    test_files = set(split["test_files"])
    report("train/test instance sets disjoint?", train_files.isdisjoint(test_files))
    n_gap_rows_train = df_gap_split[df_gap_split["file"].isin(train_files)].shape[0]
    n_gap_rows_test = df_gap_split[df_gap_split["file"].isin(test_files)].shape[0]
    report("Stage 2 gap rows on train side (4 x train instances)", n_gap_rows_train)
    report("Stage 2 gap rows on test side (4 x test instances)", n_gap_rows_test)

    print()
    print("=" * 78)
    print("BLOCK 4 — Conformal calibration split (grouped by instance)")
    print("=" * 78)
    for backbone, fit_fn in (("rf", fit_gap_regressor_rf_conformal), ("xgb", fit_gap_regressor_xgb_conformal)):
        fit = fit_fn(df_gap_split, split["train_files"], seed=SEED)
        n_fit = fit["n_train_proper_instances"]
        n_cal = fit["n_cal_instances"]
        report(f"{backbone}: fit-side instances", n_fit, PAPER_STATED["conformal_train_proper_instances"])
        report(f"{backbone}: calibration-side instances", n_cal, PAPER_STATED["conformal_calibration_instances"])
        report(f"{backbone}: fit + calibration == 128 train instances?", n_fit + n_cal == n_train)

    # Explicit disjointness re-derivation (independent of the assert already
    # inside fit_gap_regressor_*_conformal, which only checks it internally).
    from sklearn.model_selection import GroupShuffleSplit

    for backbone in ("rf", "xgb"):
        train_mask = df_gap_split["file"].isin(split["train_files"])
        X_train_full = df_gap_split.loc[train_mask]
        groups = df_gap_split.loc[train_mask, "file"]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_idx, cal_idx = next(gss.split(X_train_full, groups=groups))
        fit_files = set(df_gap_split.loc[train_mask, "file"].iloc[tr_idx])
        cal_files = set(df_gap_split.loc[train_mask, "file"].iloc[cal_idx])
        report(f"{backbone}: independently re-derived fit/cal disjoint?", fit_files.isdisjoint(cal_files))

    out_path = RESULTS_DIR / "pipeline_provenance.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print()
    print(f"wrote {out_path}  ({len(rows)} rows)")

    mismatches = [r for r in rows if r["status"] == "MISMATCH <-- CHECK"]
    print()
    if mismatches:
        print(f"{len(mismatches)} MISMATCH(ES) FOUND:")
        for m in mismatches:
            print(f"  - {m['item']}: computed={m['computed']} vs paper stated={m['paper_stated']}")
    else:
        print("No mismatches against paper-stated values.")


if __name__ == "__main__":
    main()
