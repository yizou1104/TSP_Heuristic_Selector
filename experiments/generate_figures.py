"""Regenerate the seven Results figures against the CURRENT (post S2.1
normalization) dataset and models, at seed=1, matching the paper's existing
figure structure and captions exactly (7 separate figures, same filenames).

No plotting code for the originals was found anywhere in this repository —
these are newly written to reproduce what each caption/surrounding prose
describes, using the same fitted objects and split as the rest of the
revision (prepare_stage1_data, evaluate_stage1_cv, run_integrated_pipeline).

Output: 7 PNGs at 300 dpi, written to the repository root (alongside
"revised submission.tex"), using the EXACT filenames the .tex file already
\includegraphics's, so they can be dropped into the Overleaf project root
to directly replace the stale versions:

    fig1_lr_feature_influence.png
    fig1_lr_coefficients_by_class.png
    fig2_rf_feature_weights.png
    fig3_xgb_feature_weights.png
    fig4_rf_quantile_regression.png
    fig5_rf_conformal_prediction.png
    fig6_xgb_quantile_regression.png
    fig7_xgb_conformal_prediction.png

Run from repo root: python -m experiments.generate_figures
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import STAGE1_INSTANCE_FEATURES
from src.models import (
    SEED,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    build_xgboost_tuned_pipeline,
    evaluate_stage1_cv,
    fit_logistic_regression,
    fit_xgboost_tuned,
    gini_feature_importance,
    prepare_stage1_data,
    run_integrated_pipeline,
)

OUT_DIR = Path(__file__).resolve().parents[1]  # repo root, next to the .tex
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "Dataset.csv"

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def load_df() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


# ---------------------------------------------------------------------------
# Figure 1a/1b — Logistic Regression
# ---------------------------------------------------------------------------
def make_fig1(df: pd.DataFrame) -> None:
    from src.features import STAGE1_NUMERIC_FEATURES

    # LogisticRegression has no feature_importances_, so evaluate_stage1_cv
    # (which unconditionally computes Gini importance) cannot be reused here.
    # Fit directly on all 160 instances instead, matching how the paper's LR
    # subsection reports coefficients (refit on the full data after CV).
    _, best_rows = prepare_stage1_data(df)
    X = best_rows[STAGE1_INSTANCE_FEATURES]
    y = best_rows["heuristic"]
    pipeline = fit_logistic_regression(X, y, seed=SEED)
    classes = pipeline.named_steps["classifier"].classes_
    coefs = pipeline.named_steps["classifier"].coef_  # (n_classes, n_numeric_features) -- OHE cat appended too
    n_numeric = len(STAGE1_NUMERIC_FEATURES)
    coefs_numeric = coefs[:, :n_numeric]  # standardized numeric coefficients only

    # 1a: overall feature influence = mean absolute standardized coefficient across classes
    influence = np.abs(coefs_numeric).mean(axis=0)
    order = np.argsort(influence)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(
        [STAGE1_NUMERIC_FEATURES[i] for i in order][::-1],
        [influence[i] for i in order][::-1],
        color="#4C72B0",
    )
    ax.set_xlabel("Mean |standardized coefficient| across classes")
    ax.set_title("Logistic Regression: Feature Influence")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_lr_feature_influence.png", dpi=300)
    plt.close(fig)

    # 1b: coefficients broken out by class (signed)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_numeric)
    width = 0.8 / len(classes)
    for i, cls in enumerate(classes):
        ax.bar(x + i * width, coefs_numeric[i], width=width, label=cls)
    ax.set_xticks(x + width * (len(classes) - 1) / 2)
    ax.set_xticklabels(STAGE1_NUMERIC_FEATURES, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("Logistic Regression: Coefficients by Class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_lr_coefficients_by_class.png", dpi=300)
    plt.close(fig)
    print("wrote fig1_lr_feature_influence.png, fig1_lr_coefficients_by_class.png")


# ---------------------------------------------------------------------------
# Figure 2 — Random Forest Gini feature weights
# ---------------------------------------------------------------------------
def make_fig2(df: pd.DataFrame) -> None:
    result = evaluate_stage1_cv(df, build_random_forest_pipeline, seed=SEED)
    gini = result["gini_importance"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(gini["feature"][::-1], gini["importance"][::-1], color="#55A868")
    ax.set_xlabel("Gini importance")
    ax.set_title("Random Forest: Feature Weights")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_rf_feature_weights.png", dpi=300)
    plt.close(fig)
    print("wrote fig2_rf_feature_weights.png")


# ---------------------------------------------------------------------------
# Figure 3 — XGBoost (tuned) Gini feature weights
# ---------------------------------------------------------------------------
def make_fig3(df: pd.DataFrame) -> None:
    result = evaluate_stage1_cv(
        df, build_xgboost_tuned_pipeline, seed=SEED, needs_label_encoding=True
    )
    gini = result["gini_importance"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(gini["feature"][::-1], gini["importance"][::-1], color="#C44E52")
    ax.set_xlabel("Gini importance")
    ax.set_title("XGBoost (Tuned): Feature Weights")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_xgb_feature_weights.png", dpi=300)
    plt.close(fig)
    print("wrote fig3_xgb_feature_weights.png")


# ---------------------------------------------------------------------------
# Figures 4-7 — Stage 2 integrated pipelines (Seed 1)
# ---------------------------------------------------------------------------
_PIPELINE_SPECS = {
    "fig4_rf_quantile_regression.png": ("rf", "quantile", "Random Forest + Quantile Regression (Seed 1)"),
    "fig5_rf_conformal_prediction.png": ("rf", "conformal", "Random Forest + Conformal Prediction (Seed 1)"),
    "fig6_xgb_quantile_regression.png": ("xgb", "quantile", "XGBoost + Quantile Regression (Seed 1)"),
    "fig7_xgb_conformal_prediction.png": ("xgb", "conformal", "XGBoost + Conformal Prediction (Seed 1)"),
}


def make_stage2_figure(df: pd.DataFrame, filename: str) -> None:
    backbone, method, title = _PIPELINE_SPECS[filename]
    out = run_integrated_pipeline(df, backbone=backbone, uncertainty_method=method, seed=SEED)
    results = out["results"].dropna(subset=["true_gap"]).sort_values("true_gap").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.arange(len(results))
    covered = results["in_interval"]

    yerr_lower = (results["predicted_gap"] - results["lower_gap"]).clip(lower=0)
    yerr_upper = (results["upper_gap"] - results["predicted_gap"]).clip(lower=0)
    ax.errorbar(
        x,
        results["predicted_gap"],
        yerr=[yerr_lower, yerr_upper],
        fmt="none",
        ecolor="#8172B2",
        alpha=0.5,
        capsize=2,
        label="90% prediction interval",
    )
    ax.scatter(x[covered], results.loc[covered, "true_gap"], color="#2A9D8F", s=28, label="True gap (covered)", zorder=3)
    ax.scatter(x[~covered], results.loc[~covered, "true_gap"], color="#D62839", marker="x", s=45, label="True gap (uncovered)", zorder=3)
    ax.scatter(x, results["predicted_gap"], color="black", s=14, marker="_", label="Predicted gap", zorder=4)

    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Test instance (sorted by true optimality gap)")
    ax.set_ylabel("Optimality gap (%)")
    coverage_pct = 100.0 * covered.mean()
    ax.set_title(f"{title}\nCoverage: {coverage_pct:.2f}%  |  MAE: {out['gap_mae']:.2f}pp  |  R²(log): {out['r2_log']:.4f}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=300)
    plt.close(fig)
    print(f"wrote {filename}  (coverage={coverage_pct:.2f}%, mae={out['gap_mae']:.2f}, r2_log={out['r2_log']:.4f})")


def main() -> None:
    df = load_df()
    make_fig1(df)
    make_fig2(df)
    make_fig3(df)
    for filename in _PIPELINE_SPECS:
        make_stage2_figure(df, filename)
    print("\nAll 7 figures written to:", OUT_DIR)


if __name__ == "__main__":
    main()
