"""m6 + m7 — one 1x2 figure: interval-width distribution and coverage calibration.

LEFT  : boxplot of per-instance prediction-interval widths for all four
        pipelines at seed=1, log-scaled y (widths span orders of magnitude).
RIGHT : empirical coverage against the nominal 90% target, one point per seed
        per pipeline across the 30 repeated hold-out partitions, with the 90%
        reference line drawn.

Deliberately NOT a multi-nominal-level calibration curve: producing one would
require retraining the quantile regressors at other alpha values, which would
perturb every published interval number. The across-seed view at the single
90% level visualizes the coverage-validity result already established by the
significance testing.

Output: figures/fig_intervals_calibration.png (300 dpi)

Run from repo root: python -m experiments.interval_calibration_figure
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

from src.models import SEED, run_integrated_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "Dataset.csv"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

NOMINAL = 0.90

PIPELINES = [
    ("rf", "quantile", "RF+Quantile"),
    ("rf", "conformal", "RF+Conformal"),
    ("xgb", "quantile", "XGB+Quantile"),
    ("xgb", "conformal", "XGB+Conformal"),
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


def main() -> None:
    df = pd.read_csv(DATASET_PATH)

    # ---- LEFT: interval widths at seed 1 --------------------------------
    widths, labels = [], []
    for backbone, method, label in PIPELINES:
        out = run_integrated_pipeline(df, backbone=backbone, uncertainty_method=method, seed=SEED)
        w = out["results"].dropna(subset=["true_gap"])["interval_width"].values
        widths.append(w)
        labels.append(label)
        print(f"{label:15s} median width {np.median(w):8.2f}pp   mean {np.mean(w):10.2f}pp   max {np.max(w):10.2f}pp")

    # ---- RIGHT: coverage across 30 seeds --------------------------------
    ms = pd.read_csv(RESULTS_DIR / "multi_seed_holdout.csv")
    assert len(ms) == 120, f"expected 120 rows (30 seeds x 4 pipelines), got {len(ms)}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # LEFT panel
    bp = axes[0].boxplot(widths, labels=labels, patch_artist=True, showfliers=True,
                         medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Prediction interval width (pp, log scale)")
    axes[0].set_title("Interval widths on the 32 test instances (Seed 1)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.3)

    # RIGHT panel
    for i, (backbone, method, label) in enumerate(PIPELINES):
        key = label.replace("+", "+")
        sub = ms[ms["pipeline"] == key]
        if sub.empty:
            raise SystemExit(f"no rows in multi_seed_holdout.csv for pipeline '{key}'")
        x = np.full(len(sub), i) + np.random.RandomState(0).uniform(-0.16, 0.16, len(sub))
        axes[1].scatter(x, 100 * sub["coverage"], color=COLORS[i], alpha=0.65, s=26, edgecolors="none")
        axes[1].hlines(100 * sub["coverage"].mean(), i - 0.28, i + 0.28,
                       color=COLORS[i], linewidth=2.5, zorder=3)
        print(f"{label:15s} mean coverage across 30 seeds: {100*sub['coverage'].mean():.2f}%")

    axes[1].axhline(100 * NOMINAL, color="black", linestyle="--", linewidth=1.4,
                    label=f"Nominal {100*NOMINAL:.0f}%")
    axes[1].set_xticks(range(len(PIPELINES)))
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("Empirical coverage (%)")
    axes[1].set_title("Coverage vs. nominal 90% across 30 partitions")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_intervals_calibration.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
