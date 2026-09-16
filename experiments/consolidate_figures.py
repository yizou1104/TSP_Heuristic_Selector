"""S3.3a — consolidate the 7 Results figures into 3, to recover page budget
against the 20-page limit (no appendix allowed).

Reuses the exact same evaluate_stage1_cv() / run_integrated_pipeline() calls
at seed=1 as experiments/generate_figures.py — no refitting with different
settings, no new modeling logic.

Output (300 dpi, into a new figures/ directory):
  figures/fig_stage1_importance.png : RF and XGB Gini importance, 1x2 panel
    (replaces the old fig2_rf_feature_weights.png + fig3_xgb_feature_weights.png)
  figures/fig_stage2_pipelines.png  : all 4 integrated pipelines, 2x2 panel,
    shared y-axis scale (replaces fig4/fig5/fig6/fig7)

The two LR figures (fig1_lr_feature_influence, fig1_lr_coefficients_by_class)
are dropped entirely per the consolidation spec; LR's top coefficients are
summarized in one sentence of prose in the paper instead.

Run from repo root: python -m experiments.consolidate_figures
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

from src.models import (
    SEED,
    build_random_forest_pipeline,
    build_xgboost_tuned_pipeline,
    evaluate_stage1_cv,
    run_integrated_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "Dataset.csv"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def load_df() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


# ---------------------------------------------------------------------------
# fig_stage1_importance.png — RF + XGB Gini importance, side by side
# ---------------------------------------------------------------------------
def make_fig_stage1_importance(df: pd.DataFrame) -> None:
    rf = evaluate_stage1_cv(df, build_random_forest_pipeline, seed=SEED)
    xgb = evaluate_stage1_cv(
        df, build_xgboost_tuned_pipeline, seed=SEED, needs_label_encoding=True
    )
    rf_gini = rf["gini_importance"]
    xgb_gini = xgb["gini_importance"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    axes[0].barh(rf_gini["feature"][::-1], rf_gini["importance"][::-1], color="#55A868")
    axes[0].set_xlabel("Gini importance")
    axes[0].set_title("Random Forest")

    axes[1].barh(xgb_gini["feature"][::-1], xgb_gini["importance"][::-1], color="#C44E52")
    axes[1].set_xlabel("Gini importance")
    axes[1].set_title("XGBoost (Tuned)")

    fig.suptitle("Stage 1: Feature Importance by Classifier")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_stage1_importance.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# fig_stage2_pipelines.png — all 4 pipelines, 2x2 panel, shared y-axis
# ---------------------------------------------------------------------------
_PANEL_SPECS = [
    ("rf", "quantile", "RF + Quantile"),
    ("rf", "conformal", "RF + Conformal"),
    ("xgb", "quantile", "XGB + Quantile"),
    ("xgb", "conformal", "XGB + Conformal"),
]


def make_fig_stage2_pipelines(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=False, sharey=True)
    axes = axes.flatten()

    # Determine a common y-axis range across all 4 panels first.
    all_outs = {}
    global_max = 0.0
    global_min = np.inf
    for backbone, method, _ in _PANEL_SPECS:
        out = run_integrated_pipeline(df, backbone=backbone, uncertainty_method=method, seed=SEED)
        all_outs[(backbone, method)] = out
        r = out["results"].dropna(subset=["true_gap"])
        global_max = max(global_max, r["upper_gap"].max(), r["true_gap"].max())
        global_min = min(global_min, max(r[["true_gap"]].min().iloc[0], 0.5))

    for ax, (backbone, method, title) in zip(axes, _PANEL_SPECS):
        out = all_outs[(backbone, method)]
        results = out["results"].dropna(subset=["true_gap"]).sort_values("true_gap").reset_index(drop=True)
        x = np.arange(len(results))
        covered = results["in_interval"]

        yerr_lower = (results["predicted_gap"] - results["lower_gap"]).clip(lower=0)
        yerr_upper = (results["upper_gap"] - results["predicted_gap"]).clip(lower=0)
        ax.errorbar(
            x, results["predicted_gap"], yerr=[yerr_lower, yerr_upper],
            fmt="none", ecolor="#8172B2", alpha=0.5, capsize=2,
        )
        ax.scatter(x[covered], results.loc[covered, "true_gap"], color="#2A9D8F", s=22, zorder=3)
        ax.scatter(x[~covered], results.loc[~covered, "true_gap"], color="#D62839", marker="x", s=38, zorder=3)
        ax.scatter(x, results["predicted_gap"], color="black", s=10, marker="_", zorder=4)

        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(bottom=global_min * 0.5, top=global_max * 1.3)
        coverage_pct = 100.0 * covered.mean()
        ax.set_title(
            f"{title}\nCoverage {coverage_pct:.2f}%  |  MAE {out['gap_mae']:.2f}pp  |  R²(log) {out['r2_log']:.4f}",
            fontsize=9,
        )
        ax.set_xlabel("Test instance (sorted by true gap)")

    axes[0].set_ylabel("Optimality gap (%)")
    axes[2].set_ylabel("Optimality gap (%)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2A9D8F", markersize=7, label="True gap (covered)"),
        plt.Line2D([0], [0], marker="x", color="#D62839", markersize=7, linestyle="None", label="True gap (uncovered)"),
        plt.Line2D([0], [0], marker="_", color="black", markersize=10, linestyle="None", label="Predicted gap"),
        plt.Line2D([0], [0], color="#8172B2", alpha=0.5, linewidth=4, label="90% prediction interval"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Stage 2: Integrated Pipeline Predictions on the 32 Test Instances (Seed 1)")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = FIGURES_DIR / "fig_stage2_pipelines.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    for (backbone, method, title), out in zip(_PANEL_SPECS, all_outs.values()):
        cov = 100.0 * out["results"].dropna(subset=["true_gap"])["in_interval"].mean()
        print(f"  {title}: coverage={cov:.2f}%, mae={out['gap_mae']:.2f}, r2_log={out['r2_log']:.4f}")


def main() -> None:
    df = load_df()
    make_fig_stage1_importance(df)
    make_fig_stage2_pipelines(df)
    print("\nAll consolidated figures written to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
