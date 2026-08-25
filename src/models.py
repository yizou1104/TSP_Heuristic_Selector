"""Stage 1 (best-heuristic classifier) and Stage 2 (optimality-gap regressor)
models, reconstructed from the notebook's final "K-Fold Stratified Models"
section (original lines ~6768-10592 — the last major section in the file,
appearing after several earlier, superseded single-split drafts at lines
~2440-6767).

Stage 1 models (final, tuned hyperparameters — see MODEL_SPEC_NOTES.md-style
citations in each function's docstring):
    fit_logistic_regression   <- original lines ~10341-10440 (K-Fold LR)
    fit_random_forest         <- original lines ~6775-6863   (K-Fold RF)
    fit_xgboost_tuned         <- original lines ~9356-9463   ("##XGBoost Tuned")

Stage 2 models (4 pipelines, RF/XGB backbones x Quantile/Conformal uncertainty):
    fit_gap_regressor_rf_quantile   <- original lines ~7093-7242 (K-Fold RF+Quantile)
    fit_gap_regressor_rf_conformal  <- original lines ~7576-7737 (K-Fold RF+Conformal)
    fit_gap_regressor_xgb_quantile  <- original lines ~10038-10165 (XGBoost Tuned + Quantile)
    fit_gap_regressor_xgb_conformal <- original lines ~9675-9805  (XGBoost Tuned + Conformal)

IMPORTANT: `mst_weight` is already normalised once in dataset.csv (see
src/features.py / src/build_dataset.py). Nothing in this module divides it
by `n * mean_edge_weight` again — matching the explicit fix/comment in the
notebook's final tuned sections (original lines ~9658-9659, ~10021-10022).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:  # pragma: no cover - optional dependency
    RandomForestQuantileRegressor = None

from .build_dataset import RUNTIME_CAP_MS
from .features import (
    STAGE1_CATEGORICAL_FEATURES,
    STAGE1_INSTANCE_FEATURES,
    STAGE1_NUMERIC_FEATURES,
    STAGE2_CATEGORICAL_FEATURES,
    STAGE2_FEATURE_COLUMNS,
)

SEED = 1
ALPHA = 0.1  # 90% prediction intervals throughout Stage 2


# ---------------------------------------------------------------------------
# Shared data prep
# ---------------------------------------------------------------------------
def prepare_stage1_data(df: pd.DataFrame, runtime_cap_ms: float = RUNTIME_CAP_MS):
    """Return (df_with_feasibility_cols, best_rows).

    `best_rows` has one row per instance: the row of the single feasible
    heuristic with the lowest cost (i.e. the best-heuristic label), with
    that heuristic's own feature values attached (features are identical
    across the 4 heuristic-rows of one instance).
    """
    df = df.copy()
    df["feasible"] = df["runtime_ms"] <= runtime_cap_ms
    df["best_feasible_cost"] = df[df["feasible"]].groupby("file")["cost"].transform("min")

    feasible_rows = df[df["feasible"] & df["cost"].notna()]
    best_idx = feasible_rows.groupby("file")["cost"].idxmin()
    best_rows = df.loc[best_idx].reset_index(drop=True)
    return df, best_rows


def prepare_stage2_split(df: pd.DataFrame, seed: int = SEED, test_size: float = 0.2):
    """Shared fixed 80/20 INSTANCE-level split used by both stages, plus the
    log-gap training table. Matches original lines ~7110-7130 /
    ~9693-9698.
    """
    df, best_rows = prepare_stage1_data(df)

    all_files = best_rows["file"].values
    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=seed)

    train_mask = best_rows["file"].isin(train_files)
    test_mask = best_rows["file"].isin(test_files)
    best_rows_train = best_rows[train_mask].reset_index(drop=True)
    best_rows_test = best_rows[test_mask].reset_index(drop=True)

    df_gap = df.dropna(subset=["optimality_gap"]).copy()
    df_gap["gap_log"] = np.log1p(df_gap["optimality_gap"])

    return {
        "df": df,
        "best_rows": best_rows,
        "train_files": train_files,
        "test_files": test_files,
        "best_rows_train": best_rows_train,
        "best_rows_test": best_rows_test,
        "df_gap": df_gap,
    }


# ---------------------------------------------------------------------------
# Stage 1 — pipeline builders (unfitted; used both directly and inside CV)
# ---------------------------------------------------------------------------
def build_logistic_regression_pipeline(seed: int = SEED) -> Pipeline:
    """LR spec: multiclass, L2, C=1.0, solver='lbfgs', max_iter=5000, inside
    a StandardScaler(numeric) + OneHotEncoder(categorical) pipeline.
    Original lines ~10426-10440.
    """
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), STAGE1_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), STAGE1_CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000, n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("classifier", model)])


def build_random_forest_pipeline(seed: int = SEED) -> Pipeline:
    """RF spec: n_estimators=400, max_depth=None, min_samples_leaf=2,
    n_jobs=-1, passthrough numeric + OHE categorical. Original lines
    ~6849-6863 (found verbatim, matches spec exactly).
    """
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", STAGE1_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), STAGE1_CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    return Pipeline([("preprocess", preprocess), ("classifier", model)])


def build_xgboost_tuned_pipeline(seed: int = SEED) -> Pipeline:
    """XGBoost TUNED spec: objective='multi:softprob', eval_metric='mlogloss',
    n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0. Original lines ~9435-9458 (found
    verbatim under "##XGBoost Tuned", matches spec exactly — this replaces
    an earlier, superseded n_estimators=500/max_depth=6 draft that appears
    repeatedly earlier in the notebook, e.g. original lines ~3066-3067,
    ~4276-4277, ~8102-8103).

    The caller MUST label-encode `y` (via a LabelEncoder fit on ALL class
    labels before any CV split — see `fit_xgboost_tuned`) before fitting or
    cross-validating this pipeline.
    """
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", STAGE1_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), STAGE1_CATEGORICAL_FEATURES),
        ]
    )
    model = XGBClassifier(
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
    )
    return Pipeline([("preprocess", preprocess), ("classifier", model)])


# ---------------------------------------------------------------------------
# Stage 1 — fit convenience wrappers
# ---------------------------------------------------------------------------
def fit_logistic_regression(X: pd.DataFrame, y, seed: int = SEED) -> Pipeline:
    pipeline = build_logistic_regression_pipeline(seed)
    pipeline.fit(X, y)
    return pipeline


def fit_random_forest(X: pd.DataFrame, y, seed: int = SEED) -> Pipeline:
    pipeline = build_random_forest_pipeline(seed)
    pipeline.fit(X, y)
    return pipeline


def fit_xgboost_tuned(X: pd.DataFrame, y, seed: int = SEED):
    """Fits the tuned XGBoost classifier. `y` may be raw string labels — a
    LabelEncoder is fit here and returned alongside the pipeline so callers
    can invert predictions back to heuristic names.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    pipeline = build_xgboost_tuned_pipeline(seed)
    pipeline.fit(X, y_encoded)
    return pipeline, label_encoder


# ---------------------------------------------------------------------------
# Stage 1 — stratified 5-fold CV evaluation
# ---------------------------------------------------------------------------
def evaluate_stage1_cv(
    df: pd.DataFrame,
    build_pipeline_fn: Callable[[int], Pipeline],
    seed: int = SEED,
    n_splits: int = 5,
    needs_label_encoding: bool = False,
    n_permutation_repeats: int = 30,
) -> dict:
    """Stratified 5-fold CV over all 160 instances, matching original lines
    ~6866-7025 (RF) / ~9465-9594 (XGBoost Tuned) / ~10442-10465 (LR).

    `build_pipeline_fn` is one of `build_logistic_regression_pipeline`,
    `build_random_forest_pipeline`, or `build_xgboost_tuned_pipeline`.
    `needs_label_encoding=True` for the XGBoost pipeline (LabelEncoder fit
    on ALL labels before the CV loop, per spec).

    Returns fold accuracies, out-of-fold predictions, regret stats,
    catastrophic-failure rate, Gini importance, and 30-repeat permutation
    importance.
    """
    df, best_rows = prepare_stage1_data(df)
    X = best_rows[STAGE1_INSTANCE_FEATURES]
    y_true = best_rows["heuristic"]

    label_encoder = None
    if needs_label_encoding:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_true)
    else:
        y_encoded = y_true

    pipeline = build_pipeline_fn(seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cv_results = cross_validate(
        pipeline,
        X,
        y_encoded,
        cv=cv,
        scoring={"accuracy": make_scorer(accuracy_score)},
        return_train_score=True,
        n_jobs=-1,
    )
    fold_accuracy = cv_results["test_accuracy"]
    train_accuracy = cv_results["train_accuracy"]

    y_oof_encoded = cross_val_predict(pipeline, X, y_encoded, cv=cv)
    y_oof = label_encoder.inverse_transform(y_oof_encoded) if label_encoder is not None else y_oof_encoded

    # Regret / exact-match / catastrophic-failure, using out-of-fold predictions
    pred_cost_lookup = (
        df[df["feasible"]][["file", "heuristic", "cost"]]
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
    valid = regret_df.dropna(subset=["regret"])

    # Feature importance: refit on all data
    pipeline.fit(X, y_encoded)
    gini_df = gini_feature_importance(pipeline)
    perm_df = permutation_feature_importance(
        pipeline, X, y_encoded, seed=seed, n_repeats=n_permutation_repeats
    )

    return {
        "fold_accuracy": fold_accuracy,
        "mean_accuracy": float(fold_accuracy.mean()),
        "std_accuracy": float(fold_accuracy.std()),
        "train_accuracy_mean": float(train_accuracy.mean()),
        "classification_report": classification_report(y_true, y_oof, zero_division=0),
        "oof_predictions": regret_df,
        "exact_match_rate": float((valid["true_heuristic"] == valid["predicted_heuristic"]).mean()),
        "mean_regret": float(valid["regret"].mean()),
        "median_regret": float(valid["regret"].median()),
        "catastrophic_failure_rate": float((valid["regret"] > 1).mean()),
        "gini_importance": gini_df,
        "permutation_importance": perm_df,
        "fitted_pipeline": pipeline,
        "label_encoder": label_encoder,
    }


# ---------------------------------------------------------------------------
# Feature importance helpers
# ---------------------------------------------------------------------------
def gini_feature_importance(fitted_pipeline: Pipeline) -> pd.DataFrame:
    """Gini importance from a fitted Stage-1 classifier pipeline (post
    one-hot-encoding feature names). Original lines ~6968-6984 / ~9565-9579.
    """
    cat_names = (
        fitted_pipeline.named_steps["preprocess"]
        .named_transformers_["cat"]
        .get_feature_names_out(STAGE1_CATEGORICAL_FEATURES)
    )
    feature_names = np.concatenate([STAGE1_NUMERIC_FEATURES, cat_names])
    importances = fitted_pipeline.named_steps["classifier"].feature_importances_
    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )


def permutation_feature_importance(
    fitted_pipeline: Pipeline, X: pd.DataFrame, y, seed: int = SEED, n_repeats: int = 30
) -> pd.DataFrame:
    """Permutation importance on the ORIGINAL (pre-encoding) features —
    robust to one-hot correlation. Original lines ~6986-7000 / ~9581-9594.
    """
    perm = permutation_importance(fitted_pipeline, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1)
    return pd.DataFrame(
        {
            "feature": STAGE1_INSTANCE_FEATURES,
            "importance": perm.importances_mean,
            "std": perm.importances_std,
        }
    ).sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# Stage 2 — gap-regressor preprocessing (shared by all 4 backbones)
# ---------------------------------------------------------------------------
def _build_gap_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), STAGE1_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), STAGE2_CATEGORICAL_FEATURES),
        ]
    )


def _gap_feature_importance(preprocess: ColumnTransformer, importances: np.ndarray) -> pd.DataFrame:
    cat_names = preprocess.named_transformers_["cat"].get_feature_names_out(STAGE2_CATEGORICAL_FEATURES)
    feature_names = np.concatenate([STAGE1_NUMERIC_FEATURES, cat_names])
    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )


def _gap_predict_row(instance_features: dict, heuristic: str) -> pd.DataFrame:
    row = dict(instance_features)
    row["heuristic"] = heuristic
    return pd.DataFrame([row])[STAGE2_FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Stage 2 — RF + Quantile (RandomForestQuantileRegressor)
# ---------------------------------------------------------------------------
def fit_gap_regressor_rf_quantile(df_gap: pd.DataFrame, train_files, seed: int = SEED) -> dict:
    """Original lines ~7220-7242. quantile_forest.RandomForestQuantileRegressor,
    n_estimators=200, max_depth=None, min_samples_leaf=2, predicting
    q=[0.05, 0.5, 0.95] in log space, back-transformed via expm1.
    """
    if RandomForestQuantileRegressor is None:
        raise ImportError("quantile-forest is required for fit_gap_regressor_rf_quantile")

    train_mask = df_gap["file"].isin(train_files)
    X_train = df_gap.loc[train_mask, STAGE2_FEATURE_COLUMNS]
    y_train = df_gap.loc[train_mask, "gap_log"]

    preprocess = _build_gap_preprocessor()
    X_train_proc = preprocess.fit_transform(X_train)

    model = RandomForestQuantileRegressor(
        n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    model.fit(X_train_proc, y_train)

    def predict(instance_features: dict, heuristic: str):
        X = preprocess.transform(_gap_predict_row(instance_features, heuristic))
        q = model.predict(X, quantiles=[0.05, 0.5, 0.95])
        lower = np.expm1(max(q[0, 0], 0.0))
        point = np.expm1(q[0, 1])
        upper = np.expm1(q[0, 2])
        return point, lower, upper

    return {
        "backbone": "rf",
        "uncertainty_method": "quantile",
        "preprocess": preprocess,
        "model": model,
        "predict": predict,
        "feature_importance": _gap_feature_importance(preprocess, model.feature_importances_),
    }


# ---------------------------------------------------------------------------
# Stage 2 — RF + Conformal
# ---------------------------------------------------------------------------
def fit_gap_regressor_rf_conformal(df_gap: pd.DataFrame, train_files, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Original lines ~7686-7737. RandomForestRegressor(squared error),
    n_estimators=200, max_depth=None, min_samples_leaf=2. Calibration split
    is INSTANCE-GROUPED (S2.2 fix): GroupShuffleSplit on the "file" column,
    20% of TRAINING INSTANCES held out for calibration, so all heuristic-rows
    of a given instance stay together on one side of the split. This
    preserves the exchangeability assumption conformal prediction relies on
    for its coverage guarantee (a plain row-level train_test_split could put
    different heuristic-rows of the SAME instance into both train and
    calibration). Conformal score = 90th percentile of absolute calibration
    residuals in log space, applied symmetrically.
    """
    train_mask = df_gap["file"].isin(train_files)
    X_train_full = df_gap.loc[train_mask, STAGE2_FEATURE_COLUMNS]
    y_train_full = df_gap.loc[train_mask, "gap_log"]

    groups = df_gap.loc[train_mask, "file"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, cal_idx = next(gss.split(X_train_full, y_train_full, groups=groups))
    X_train_proper, X_cal = X_train_full.iloc[train_idx], X_train_full.iloc[cal_idx]
    y_train_proper, y_cal = y_train_full.iloc[train_idx], y_train_full.iloc[cal_idx]

    train_proper_files = set(df_gap.loc[train_mask, "file"].iloc[train_idx])
    cal_files = set(df_gap.loc[train_mask, "file"].iloc[cal_idx])
    assert train_proper_files.isdisjoint(cal_files), "Instance leakage between train and calibration!"

    preprocess = _build_gap_preprocessor()
    model = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    pipeline = Pipeline([("preprocess", preprocess), ("regressor", model)])
    pipeline.fit(X_train_proper, y_train_proper)

    y_cal_pred_log = pipeline.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred_log)
    conformal_score = float(np.quantile(residuals, 1 - alpha))

    def predict(instance_features: dict, heuristic: str):
        X = _gap_predict_row(instance_features, heuristic)
        pred_log = pipeline.predict(X)[0]
        point = np.expm1(pred_log)
        lower = np.expm1(max(pred_log - conformal_score, 0.0))
        upper = np.expm1(pred_log + conformal_score)
        return point, lower, upper

    return {
        "backbone": "rf",
        "uncertainty_method": "conformal",
        "pipeline": pipeline,
        "conformal_score": conformal_score,
        "predict": predict,
        "feature_importance": _gap_feature_importance(
            pipeline.named_steps["preprocess"], pipeline.named_steps["regressor"].feature_importances_
        ),
        "n_train_proper_instances": len(train_proper_files),
        "n_cal_instances": len(cal_files),
    }


# ---------------------------------------------------------------------------
# Stage 2 — XGB + Quantile (3 separate quantile-loss models)
# ---------------------------------------------------------------------------
_XGB_GAP_COMMON_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=0.1,
    n_jobs=-1,
)


def fit_gap_regressor_xgb_quantile(df_gap: pd.DataFrame, train_files, seed: int = SEED) -> dict:
    """Original lines ~10144-10165. Three XGBRegressor models
    (objective='reg:quantileerror', quantile_alpha=q) at q=0.05/0.50/0.95,
    each with n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0.1.
    """
    train_mask = df_gap["file"].isin(train_files)
    X_train = df_gap.loc[train_mask, STAGE2_FEATURE_COLUMNS]
    y_train = df_gap.loc[train_mask, "gap_log"]

    preprocess = _build_gap_preprocessor()
    X_train_proc = preprocess.fit_transform(X_train)

    quantiles = [0.05, 0.5, 0.95]
    models = {}
    for q in quantiles:
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, random_state=seed, **_XGB_GAP_COMMON_PARAMS)
        m.fit(X_train_proc, y_train)
        models[q] = m

    def predict(instance_features: dict, heuristic: str):
        X = preprocess.transform(_gap_predict_row(instance_features, heuristic))
        point = np.expm1(models[0.5].predict(X)[0])
        lower = np.expm1(models[0.05].predict(X)[0])
        upper = np.expm1(models[0.95].predict(X)[0])
        return point, lower, upper

    return {
        "backbone": "xgb",
        "uncertainty_method": "quantile",
        "preprocess": preprocess,
        "models": models,
        "predict": predict,
        "feature_importance": _gap_feature_importance(preprocess, models[0.5].feature_importances_),
    }


# ---------------------------------------------------------------------------
# Stage 2 — XGB + Conformal
# ---------------------------------------------------------------------------
def fit_gap_regressor_xgb_conformal(df_gap: pd.DataFrame, train_files, seed: int = SEED, alpha: float = ALPHA) -> dict:
    """Original lines ~9780-9805. Single XGBRegressor
    (objective='reg:squarederror'), same INSTANCE-GROUPED 20% calibration
    split as RF+Conformal (S2.2 fix — see fit_gap_regressor_rf_conformal's
    docstring), single global conformal score.
    """
    train_mask = df_gap["file"].isin(train_files)
    X_train_full = df_gap.loc[train_mask, STAGE2_FEATURE_COLUMNS]
    y_train_full = df_gap.loc[train_mask, "gap_log"]

    groups = df_gap.loc[train_mask, "file"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, cal_idx = next(gss.split(X_train_full, y_train_full, groups=groups))
    X_train_proper, X_cal = X_train_full.iloc[train_idx], X_train_full.iloc[cal_idx]
    y_train_proper, y_cal = y_train_full.iloc[train_idx], y_train_full.iloc[cal_idx]

    train_proper_files = set(df_gap.loc[train_mask, "file"].iloc[train_idx])
    cal_files = set(df_gap.loc[train_mask, "file"].iloc[cal_idx])
    assert train_proper_files.isdisjoint(cal_files), "Instance leakage between train and calibration!"

    preprocess = _build_gap_preprocessor()
    model = XGBRegressor(objective="reg:squarederror", random_state=seed, **_XGB_GAP_COMMON_PARAMS)
    pipeline = Pipeline([("preprocess", preprocess), ("regressor", model)])
    pipeline.fit(X_train_proper, y_train_proper)

    y_cal_pred_log = pipeline.predict(X_cal)
    residuals = np.abs(y_cal - y_cal_pred_log)
    conformal_score = float(np.quantile(residuals, 1 - alpha))

    def predict(instance_features: dict, heuristic: str):
        X = _gap_predict_row(instance_features, heuristic)
        pred_log = pipeline.predict(X)[0]
        point = np.expm1(pred_log)
        lower = np.expm1(max(pred_log - conformal_score, 0.0))
        upper = np.expm1(pred_log + conformal_score)
        return point, lower, upper

    return {
        "backbone": "xgb",
        "uncertainty_method": "conformal",
        "pipeline": pipeline,
        "conformal_score": conformal_score,
        "predict": predict,
        "feature_importance": _gap_feature_importance(
            pipeline.named_steps["preprocess"], pipeline.named_steps["regressor"].feature_importances_
        ),
        "n_train_proper_instances": len(train_proper_files),
        "n_cal_instances": len(cal_files),
    }


_STAGE2_FIT_FUNCTIONS = {
    ("rf", "quantile"): fit_gap_regressor_rf_quantile,
    ("rf", "conformal"): fit_gap_regressor_rf_conformal,
    ("xgb", "quantile"): fit_gap_regressor_xgb_quantile,
    ("xgb", "conformal"): fit_gap_regressor_xgb_conformal,
}


# ---------------------------------------------------------------------------
# Full integrated pipeline (Stage 1 fit+predict -> Stage 2 fit+predict -> metrics)
# ---------------------------------------------------------------------------
def run_integrated_pipeline(
    df: pd.DataFrame,
    backbone: str,
    uncertainty_method: str,
    seed: int = SEED,
    use_true_heuristic: bool = False,
) -> dict:
    """Fixed 80/20 instance-level split (SEED=1 -> 128 train / 32 test
    instances). Stage 1 classifier (RF or XGB, matching `backbone`) is
    trained on the 128 training instances. Stage 2 gap regressor is trained
    on all 4 heuristics x 128 training instances. At inference on the 32
    test instances, Stage 1's predicted heuristic is added as the 11th
    feature before Stage 2 predicts the gap.

    `backbone`: 'rf' or 'xgb'. `uncertainty_method`: 'quantile' or 'conformal'.

    `use_true_heuristic` (S2.3 ablation, default False): when False
    (default), Stage 2 is fed Stage 1's PREDICTED heuristic label
    (`pred_heuristic`) — this is the real end-to-end pipeline and exactly
    reproduces prior behavior. When True, Stage 2 is instead fed the TRUE
    best heuristic label (`true_heuristic`), simulating a perfect Stage 1 —
    an oracle scenario used to quantify how much Stage 1's classification
    errors cost Stage 2's gap-prediction accuracy (see
    experiments/stage1_ablation.py). This only changes which heuristic label
    routes into Stage 2's gap regressor; `classifier_accuracy` and the
    classification results (`true_heuristic`, `predicted_heuristic`,
    `classifier_correct`) are computed identically either way.

    Matches, depending on the combination:
      rf  + quantile  -> original lines ~7029-7513
      rf  + conformal -> original lines ~7514-8014
      xgb + quantile  -> original lines ~9986-10340 (XGBoost TUNED)
      xgb + conformal -> original lines ~9620-9985  (XGBoost TUNED)
    """
    if backbone not in ("rf", "xgb"):
        raise ValueError("backbone must be 'rf' or 'xgb'")
    if uncertainty_method not in ("quantile", "conformal"):
        raise ValueError("uncertainty_method must be 'quantile' or 'conformal'")

    split = prepare_stage2_split(df, seed=seed)
    best_rows_train = split["best_rows_train"]
    best_rows_test = split["best_rows_test"]
    df_gap = split["df_gap"]
    train_files = split["train_files"]

    X_train_class = best_rows_train[STAGE1_INSTANCE_FEATURES]
    y_train_class = best_rows_train["heuristic"]
    X_test_class = best_rows_test[STAGE1_INSTANCE_FEATURES]
    y_test_class = best_rows_test["heuristic"].values

    if backbone == "rf":
        pipeline_class = fit_random_forest(X_train_class, y_train_class, seed=seed)
        y_pred_class = pipeline_class.predict(X_test_class)
    else:
        pipeline_class, label_encoder = fit_xgboost_tuned(X_train_class, y_train_class, seed=seed)
        y_pred_class = label_encoder.inverse_transform(
            pipeline_class.predict(X_test_class)
        )

    class_accuracy = float(accuracy_score(y_test_class, y_pred_class))

    gap_fit_fn = _STAGE2_FIT_FUNCTIONS[(backbone, uncertainty_method)]
    gap_fit = gap_fit_fn(df_gap, train_files, seed=seed)
    predict_gap = gap_fit["predict"]

    df_full = split["df"]
    results = []
    for pos in range(len(best_rows_test)):
        row = best_rows_test.iloc[pos]
        file = row["file"]
        true_heuristic = y_test_class[pos]
        pred_heuristic = y_pred_class[pos]
        heuristic_for_stage2 = true_heuristic if use_true_heuristic else pred_heuristic

        instance_features = {feat: row[feat] for feat in STAGE1_INSTANCE_FEATURES}
        point, lower, upper = predict_gap(instance_features, heuristic_for_stage2)

        true_gap_row = df_gap[(df_gap["file"] == file) & (df_gap["heuristic"] == heuristic_for_stage2)]
        true_gap = true_gap_row["optimality_gap"].iloc[0] if len(true_gap_row) else np.nan

        cost_row = df_full[
            (df_full["file"] == file) & (df_full["heuristic"] == pred_heuristic) & df_full["feasible"]
        ]
        best_cost = row["best_feasible_cost"]
        regret = (cost_row["cost"].iloc[0] / best_cost - 1) if len(cost_row) else np.nan

        results.append(
            {
                "file": file,
                "true_heuristic": true_heuristic,
                "predicted_heuristic": pred_heuristic,
                "classifier_correct": true_heuristic == pred_heuristic,
                "predicted_gap": point,
                "lower_gap": lower,
                "upper_gap": upper,
                "interval_width": upper - lower,
                "true_gap": true_gap,
                "gap_error": point - true_gap if not np.isnan(true_gap) else np.nan,
                "in_interval": (not np.isnan(true_gap)) and (lower <= true_gap <= upper),
                "regret": regret,
            }
        )

    results_df = pd.DataFrame(results)
    valid = results_df.dropna(subset=["true_gap"])
    regret_all = results_df["regret"].dropna()

    gap_mae = mean_absolute_error(valid["true_gap"], valid["predicted_gap"])
    gap_rmse = float(np.sqrt(mean_squared_error(valid["true_gap"], valid["predicted_gap"])))
    r2_log = r2_score(np.log1p(valid["true_gap"]), np.log1p(valid["predicted_gap"]))

    within_tolerance = {
        t: float((np.abs(valid["gap_error"]) <= t).mean()) for t in (10, 20, 30)
    }

    return {
        "backbone": backbone,
        "uncertainty_method": uncertainty_method,
        "classifier_accuracy": class_accuracy,
        "results": results_df,
        "gap_mae": float(gap_mae),
        "gap_rmse": gap_rmse,
        "r2_log": float(r2_log),
        "coverage": float(valid["in_interval"].mean()),
        "mean_interval_width": float(valid["interval_width"].mean()),
        "median_interval_width": float(valid["interval_width"].median()),
        "within_tolerance": within_tolerance,
        "regret_mean": float(regret_all.mean()),
        "regret_median": float(regret_all.median()),
        "regret_max": float(regret_all.max()),
        "catastrophic_failure_rate": float((regret_all > 1).mean()),
        "gap_feature_importance": gap_fit["feature_importance"],
        "conformal_score": gap_fit.get("conformal_score"),
    }
