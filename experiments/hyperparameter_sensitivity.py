"""R6 — hyperparameter sensitivity grid for both Stage 1 tree selectors.

Sweeps max_depth x n_estimators for XGBoost and Random Forest, reporting
mean and std 5-fold CV accuracy at seed=1 for every cell. Everything other
than the two swept parameters is held at the paper's published values.

Light path only: uses cross_val_score directly rather than
evaluate_stage1_cv, because permutation importance (30 repeats) dominates
the runtime of the latter and is not needed for a sensitivity sweep.

The paper states XGBoost was tuned over six configurations and settled on
max_depth=3, n_estimators=100. This script reports plainly whether any grid
cell beats that configuration; it does NOT change the paper's chosen
configuration.

Output: results/hyperparameter_sensitivity.csv

Run from repo root: python -m experiments.hyperparameter_sensitivity
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (
    STAGE1_CATEGORICAL_FEATURES,
    STAGE1_INSTANCE_FEATURES,
    STAGE1_NUMERIC_FEATURES,
)
from src.models import SEED, prepare_stage1_data

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "Dataset.csv"
RESULTS_DIR = REPO_ROOT / "results"

# The paper's published configurations.
XGB_CHOSEN = {"max_depth": 3, "n_estimators": 100}
RF_CHOSEN = {"max_depth": None, "n_estimators": 400}

XGB_DEPTHS = [2, 3, 4, 5, 6]
XGB_ESTIMATORS = [50, 100, 200, 500]
RF_DEPTHS = [None, 5, 10, 20]
RF_ESTIMATORS = [100, 200, 400, 800]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", "passthrough", STAGE1_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), STAGE1_CATEGORICAL_FEATURES),
        ]
    )


def score_cell(model, X, y) -> tuple[float, float]:
    pipeline = Pipeline([("preprocess", _preprocessor()), ("classifier", model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    _, best_rows = prepare_stage1_data(df)
    X = best_rows[STAGE1_INSTANCE_FEATURES]
    y_raw = best_rows["heuristic"]
    y_encoded = LabelEncoder().fit_transform(y_raw)

    rows = []

    # ---- XGBoost grid ----------------------------------------------------
    for depth in XGB_DEPTHS:
        for n_est in XGB_ESTIMATORS:
            model = XGBClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                min_child_weight=1,
                gamma=0.0,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=SEED,
                n_jobs=-1,
            )
            mean, std = score_cell(model, X, y_encoded)
            rows.append(
                {
                    "model": "XGBoost",
                    "max_depth": "None" if depth is None else depth,
                    "n_estimators": n_est,
                    "cv_accuracy_mean": round(mean, 4),
                    "cv_accuracy_std": round(std, 4),
                    "is_paper_choice": depth == XGB_CHOSEN["max_depth"] and n_est == XGB_CHOSEN["n_estimators"],
                }
            )

    # ---- Random Forest grid ---------------------------------------------
    for depth in RF_DEPTHS:
        for n_est in RF_ESTIMATORS:
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                min_samples_leaf=2,
                random_state=SEED,
                n_jobs=-1,
            )
            mean, std = score_cell(model, X, y_raw)
            rows.append(
                {
                    "model": "RandomForest",
                    "max_depth": "None" if depth is None else depth,
                    "n_estimators": n_est,
                    "cv_accuracy_mean": round(mean, 4),
                    "cv_accuracy_std": round(std, 4),
                    "is_paper_choice": depth == RF_CHOSEN["max_depth"] and n_est == RF_CHOSEN["n_estimators"],
                }
            )

    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "hyperparameter_sensitivity.csv"
    out.to_csv(out_path, index=False)

    # ---- report ----------------------------------------------------------
    for model_name in ("XGBoost", "RandomForest"):
        sub = out[out["model"] == model_name]
        print("=" * 78)
        print(f"{model_name}: 5-fold CV accuracy (seed=1)")
        print("=" * 78)
        pivot = sub.pivot(index="max_depth", columns="n_estimators", values="cv_accuracy_mean")
        print(pivot.to_string())
        print()

        chosen = sub[sub["is_paper_choice"]]
        if chosen.empty:
            print(f"  WARNING: paper's chosen configuration not found in the {model_name} grid.")
            continue
        chosen_row = chosen.iloc[0]
        chosen_acc = chosen_row["cv_accuracy_mean"]
        best_row = sub.loc[sub["cv_accuracy_mean"].idxmax()]
        print(f"  Paper's choice: max_depth={chosen_row['max_depth']}, "
              f"n_estimators={chosen_row['n_estimators']} -> {chosen_acc:.4f} "
              f"(+/- {chosen_row['cv_accuracy_std']:.4f})")
        print(f"  Grid best:      max_depth={best_row['max_depth']}, "
              f"n_estimators={best_row['n_estimators']} -> {best_row['cv_accuracy_mean']:.4f} "
              f"(+/- {best_row['cv_accuracy_std']:.4f})")

        beaters = sub[sub["cv_accuracy_mean"] > chosen_acc]
        if len(beaters):
            print(f"  *** {len(beaters)} of {len(sub)} grid cells BEAT the paper's chosen configuration:")
            for _, r in beaters.sort_values("cv_accuracy_mean", ascending=False).iterrows():
                delta = r["cv_accuracy_mean"] - chosen_acc
                print(f"        depth={r['max_depth']}, n_est={r['n_estimators']}: "
                      f"{r['cv_accuracy_mean']:.4f} (+{delta:.4f} = +{100*delta:.2f}pp)")
        else:
            print(f"  No grid cell beats the paper's chosen configuration.")
        spread = sub["cv_accuracy_mean"].max() - sub["cv_accuracy_mean"].min()
        print(f"  Grid spread (max - min): {spread:.4f} ({100*spread:.2f}pp)")
        print(f"  Typical fold-to-fold std in this grid: {sub['cv_accuracy_std'].mean():.4f}")
        print()

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
