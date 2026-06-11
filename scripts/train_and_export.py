#!/usr/bin/env python3
"""Train all heuristic-selection and gap-prediction models, then serialize them.

Usage:
    python scripts/train_and_export.py
    python scripts/train_and_export.py --data data/Dataset.csv --out src/tsp_heuristics/data/models/

Trains:
  - Random Forest (seeds 1, 2, 3): classifier + quantile gap regressor
  - XGBoost      (seeds 1, 2, 3): classifier + three quantile gap regressors (q05, q50, q95)
  - Logistic Regression (no seed): classifier + linear gap regressor

All models saved as compressed joblib files: <name>.pkl.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

RUNTIME_CAP = 600_000
EPS = 0.05

NUMERIC_FEATURES = [
    "n",
    "min_edge_weight",
    "std_edge_weight",
    "cv_edge_weight",
    "edge_weight_skewness",
    "pct_short_edges",
    "triangle_violation_rate",
    "avg_violation_magnitude",
    "mst_weight",
]
CATEGORICAL_FEATURES = ["type"]
INSTANCE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
GAP_CATEGORICAL = ["type", "heuristic"]


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (best_rows, gap_df) prepared from Dataset.csv."""
    df = pd.read_csv(csv_path)

    df["feasible"] = df["runtime_ms"] <= RUNTIME_CAP
    df["best_feasible_cost"] = (
        df[df["feasible"]].groupby("file")["cost"].transform("min")
    )
    df["is_best"] = df["feasible"] & (
        df["cost"] <= (1 + EPS) * df["best_feasible_cost"]
    )

    best_rows = (
        df[df["is_best"]]
        .sort_values(["file", "cost"])
        .groupby("file")
        .first()
        .reset_index()
    )

    gap_df = df.dropna(subset=["optimality_gap"]).copy()
    gap_df["gap_log"] = np.log1p(gap_df["optimality_gap"])

    print(f"Instances for classification: {len(best_rows)}")
    print(f"Rows for gap regression: {len(gap_df)}")
    print(f"Heuristic distribution:\n{best_rows['heuristic'].value_counts()}")
    return best_rows, gap_df


def make_clf_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def make_gap_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), GAP_CATEGORICAL),
        ]
    )


def train_rf(
    best_rows: pd.DataFrame,
    gap_df: pd.DataFrame,
    seed: int,
    out_dir: Path,
) -> None:
    print(f"\n── Random Forest seed={seed} ──")

    X = best_rows[INSTANCE_FEATURES]
    y = best_rows["heuristic"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=seed, stratify=y_enc
    )

    clf_pipeline = Pipeline([
        ("preprocess", make_clf_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    clf_pipeline.fit(X_train, y_train)
    acc = (clf_pipeline.predict(X_test) == y_test).mean()
    print(f"  Classifier accuracy: {acc:.4f}")

    # Gap regressor — RandomForestQuantileRegressor
    X_gap = gap_df[INSTANCE_FEATURES + ["heuristic"]]
    y_gap = gap_df["gap_log"]

    unique_files = gap_df["file"].unique()
    train_files, test_files = train_test_split(unique_files, test_size=0.2, random_state=1)
    train_mask = gap_df["file"].isin(train_files)
    test_mask = gap_df["file"].isin(test_files)

    gap_prep = make_gap_preprocessor()
    X_gap_train_proc = gap_prep.fit_transform(X_gap[train_mask])
    X_gap_test_proc = gap_prep.transform(X_gap[test_mask])

    qrf = RandomForestQuantileRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    qrf.fit(X_gap_train_proc, y_gap[train_mask])

    preds = qrf.predict(X_gap_test_proc, quantiles=0.5)
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(np.expm1(y_gap[test_mask]), np.expm1(preds))
    print(f"  Gap regressor MAE: {mae:.2f}%")

    # Save
    joblib.dump(
        {"pipeline": clf_pipeline, "label_encoder": le},
        out_dir / f"rf_s{seed}_clf.pkl.gz",
        compress=("gzip", 6),
    )
    joblib.dump(
        {"preprocessor": gap_prep, "regressor": qrf},
        out_dir / f"rf_s{seed}_qrf.pkl.gz",
        compress=("gzip", 6),
    )
    print(f"  Saved rf_s{seed}_clf.pkl.gz and rf_s{seed}_qrf.pkl.gz")


def train_xgb(
    best_rows: pd.DataFrame,
    gap_df: pd.DataFrame,
    seed: int,
    out_dir: Path,
) -> None:
    print(f"\n── XGBoost seed={seed} ──")

    X = best_rows[INSTANCE_FEATURES]
    y = best_rows["heuristic"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=seed, stratify=y_enc
    )

    clf_prep = make_clf_preprocessor()
    xgb_clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        gamma=0.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    clf_pipeline = Pipeline([
        ("preprocess", clf_prep),
        ("classifier", xgb_clf),
    ])
    clf_pipeline.fit(X_train, y_train)
    acc = (clf_pipeline.predict(X_test) == y_test).mean()
    print(f"  Classifier accuracy: {acc:.4f}")

    # Gap regressors — three quantile XGB models
    X_gap = gap_df[INSTANCE_FEATURES + ["heuristic"]]
    y_gap = gap_df["gap_log"]

    unique_files = gap_df["file"].unique()
    train_files, test_files = train_test_split(unique_files, test_size=0.2, random_state=1)
    train_mask = gap_df["file"].isin(train_files)
    test_mask = gap_df["file"].isin(test_files)

    gap_prep = make_gap_preprocessor()
    X_train_proc = gap_prep.fit_transform(X_gap[train_mask])
    X_test_proc = gap_prep.transform(X_gap[test_mask])

    common = dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1, reg_alpha=0.1,
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    quantile_models: dict[str, XGBRegressor] = {}
    for q in (0.05, 0.5, 0.95):
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **common)
        m.fit(X_train_proc, y_gap[train_mask])
        quantile_models[f"q{int(q*100):02d}"] = m

    from sklearn.metrics import mean_absolute_error
    pred_med = quantile_models["q50"].predict(X_test_proc)
    mae = mean_absolute_error(np.expm1(y_gap[test_mask]), np.expm1(pred_med))
    print(f"  Gap regressor MAE (median): {mae:.2f}%")

    joblib.dump(
        {"pipeline": clf_pipeline, "label_encoder": le},
        out_dir / f"xgb_s{seed}_clf.pkl.gz",
        compress=("gzip", 6),
    )
    joblib.dump(
        {"preprocessor": gap_prep, "models": quantile_models},
        out_dir / f"xgb_s{seed}_qreg.pkl.gz",
        compress=("gzip", 6),
    )
    print(f"  Saved xgb_s{seed}_clf.pkl.gz and xgb_s{seed}_qreg.pkl.gz")


def train_lr(
    best_rows: pd.DataFrame,
    gap_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    print("\n── Logistic Regression ──")

    X = best_rows[INSTANCE_FEATURES]
    y = best_rows["heuristic"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=3, stratify=y_enc
    )

    clf_pipeline = Pipeline([
        ("preprocess", make_clf_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=5000, solver="lbfgs", n_jobs=-1
        )),
    ])
    clf_pipeline.fit(X_train, y_train)
    acc = (clf_pipeline.predict(X_test) == y_test).mean()
    print(f"  Classifier accuracy: {acc:.4f}")

    # Gap regressor — linear regression (no quantile support)
    X_gap = gap_df[INSTANCE_FEATURES + ["heuristic"]]
    y_gap = gap_df["gap_log"]

    unique_files = gap_df["file"].unique()
    train_files, _ = train_test_split(unique_files, test_size=0.2, random_state=1)
    train_mask = gap_df["file"].isin(train_files)

    gap_prep = make_gap_preprocessor()
    X_gap_train_proc = gap_prep.fit_transform(X_gap[train_mask])

    linreg = LinearRegression()
    linreg.fit(X_gap_train_proc, y_gap[train_mask])

    joblib.dump(
        {"pipeline": clf_pipeline, "label_encoder": le},
        out_dir / "lr_clf.pkl.gz",
        compress=("gzip", 6),
    )
    joblib.dump(
        {"preprocessor": gap_prep, "regressor": linreg},
        out_dir / "lr_reg.pkl.gz",
        compress=("gzip", 6),
    )
    print("  Saved lr_clf.pkl.gz and lr_reg.pkl.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/Dataset.csv",
        help="Path to Dataset.csv (default: data/Dataset.csv)",
    )
    parser.add_argument(
        "--out",
        default="src/tsp_heuristics/data/models/",
        help="Output directory for .pkl.gz files",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        sys.exit(f"Dataset not found: {data_path}")

    print(f"Loading {data_path} …")
    best_rows, gap_df = load_dataset(data_path)

    for seed in (1, 2, 3):
        train_rf(best_rows, gap_df, seed, out_dir)

    for seed in (1, 2, 3):
        train_xgb(best_rows, gap_df, seed, out_dir)

    train_lr(best_rows, gap_df, out_dir)

    print("\n✓ All models saved to", out_dir)
    sizes = sorted(out_dir.glob("*.pkl.gz"), key=lambda p: p.stat().st_size, reverse=True)
    for p in sizes:
        print(f"  {p.name:40s}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
