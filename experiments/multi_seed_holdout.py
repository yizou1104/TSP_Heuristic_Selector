"""S2.5 / reviewer Major Issue 4: repeated hold-out validation.

The manuscript's headline Stage 2 figures come from a single Seed-1 80/20
instance-level split, and that same split was also used to choose which of
the four Stage 2 pipelines to report. Those 32 test instances therefore
served as model-selection data rather than as an untouched test set, and
nothing in the single-split analysis establishes that the reported figures
are stable across different partitions of the data.

This script re-runs the complete two-stage pipeline over many independent
seeds. Each seed produces a fresh instance-level 80/20 split (and fresh
model random states), so the spread across seeds measures how much of the
single-split result was partition-specific.

It reports, for every pipeline:
  * mean +/- standard deviation of each headline metric across seeds,
  * the min/max range actually observed,
  * how often each pipeline would have been selected as the winner had that
    seed been the one used for model selection.

That last statistic speaks directly to the reviewer's concern: if
RF+Quantile wins only occasionally, the Seed-1 choice was partly luck.

Usage:
    python experiments/multi_seed_holdout.py [n_seeds]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_PATH  # noqa: E402
from src.models import run_integrated_pipeline  # noqa: E402

PIPELINES = [("rf", "quantile"), ("rf", "conformal"), ("xgb", "quantile"), ("xgb", "conformal")]

METRICS = [
    "classifier_accuracy",
    "gap_mae",
    "gap_rmse",
    "r2_log",
    "coverage",
    "median_interval_width",
    "regret_mean",
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV = RESULTS_DIR / "multi_seed_holdout.csv"
SUMMARY_CSV = RESULTS_DIR / "multi_seed_holdout_summary.csv"

NOMINAL_COVERAGE = 0.90


def label(backbone: str, uncertainty: str) -> str:
    return f"{'RF' if backbone == 'rf' else 'XGB'}+{'Quantile' if uncertainty == 'quantile' else 'Conformal'}"


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seeds = list(range(1, n_seeds + 1))

    df = pd.read_csv(DATASET_PATH)
    pd.set_option("display.width", 250)

    print(f"Running {len(PIPELINES)} pipelines x {len(seeds)} seeds "
          f"= {len(PIPELINES) * len(seeds)} full two-stage fits ...\n")

    rows = []
    for seed in seeds:
        for backbone, uncertainty in PIPELINES:
            r = run_integrated_pipeline(df, backbone, uncertainty, seed=seed)
            row = {"seed": seed, "pipeline": label(backbone, uncertainty)}
            for m in METRICS:
                row[m] = r[m]
            rows.append(row)
        print(f"  seed {seed:>3d} done")

    res = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS_CSV, index=False)

    # ---------------------------------------------------------------
    # Aggregate: mean +/- std across seeds
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"REPEATED HOLD-OUT RESULTS ACROSS {len(seeds)} SEEDS (mean +/- sd [min, max])")
    print("=" * 78)

    summary_rows = []
    for name in [label(b, u) for b, u in PIPELINES]:
        sub = res[res["pipeline"] == name]
        print(f"\n{name}")
        entry = {"pipeline": name}
        for m in METRICS:
            v = sub[m]
            print(f"  {m:22s}: {v.mean():9.4f} +/- {v.std():7.4f}   [{v.min():9.4f}, {v.max():9.4f}]")
            entry[f"{m}_mean"] = v.mean()
            entry[f"{m}_std"] = v.std()
            entry[f"{m}_min"] = v.min()
            entry[f"{m}_max"] = v.max()
        summary_rows.append(entry)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    # ---------------------------------------------------------------
    # Seed-1 vs. across-seed comparison (was Seed 1 optimistic?)
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SEED 1 VERSUS THE ACROSS-SEED DISTRIBUTION")
    print("=" * 78)
    hdr = f"{'Pipeline':16s} | {'Metric':22s} | {'Seed 1':>9s} | {'Mean':>9s} | {'Percentile':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for name in [label(b, u) for b, u in PIPELINES]:
        sub = res[res["pipeline"] == name]
        s1 = sub[sub["seed"] == 1].iloc[0]
        for m in ["gap_mae", "r2_log", "coverage"]:
            pct = (sub[m] <= s1[m]).mean() * 100
            print(f"{name:16s} | {m:22s} | {s1[m]:9.4f} | {sub[m].mean():9.4f} | {pct:9.1f}%")

    # ---------------------------------------------------------------
    # Model-selection stability (the reviewer's core concern)
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MODEL-SELECTION STABILITY: which pipeline wins on each seed?")
    print("=" * 78)
    for criterion, better in [("gap_mae", "min"), ("r2_log", "max")]:
        idx = res.groupby("seed")[criterion].idxmin() if better == "min" else res.groupby("seed")[criterion].idxmax()
        winners = res.loc[idx, "pipeline"].value_counts()
        print(f"\nBest by {criterion} ({better}):")
        for name in [label(b, u) for b, u in PIPELINES]:
            c = int(winners.get(name, 0))
            print(f"  {name:16s}: won {c:3d}/{len(seeds)} seeds ({c / len(seeds) * 100:5.1f}%)")

    # coverage closest to nominal
    res["_cov_err"] = (res["coverage"] - NOMINAL_COVERAGE).abs()
    idx = res.groupby("seed")["_cov_err"].idxmin()
    winners = res.loc[idx, "pipeline"].value_counts()
    print(f"\nCoverage closest to nominal {NOMINAL_COVERAGE:.0%}:")
    for name in [label(b, u) for b, u in PIPELINES]:
        c = int(winners.get(name, 0))
        print(f"  {name:16s}: won {c:3d}/{len(seeds)} seeds ({c / len(seeds) * 100:5.1f}%)")

    # ---------------------------------------------------------------
    # Paired comparison: RF+Quantile vs each other pipeline on MAE
    # ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PAIRED PER-SEED COMPARISON (RF+Quantile vs. others, gap_mae)")
    print("=" * 78)
    base = res[res["pipeline"] == "RF+Quantile"].set_index("seed")["gap_mae"]
    for name in ["RF+Conformal", "XGB+Quantile", "XGB+Conformal"]:
        other = res[res["pipeline"] == name].set_index("seed")["gap_mae"]
        diff = other - base            # positive => RF+Quantile better
        wins = int((diff > 0).sum())
        print(f"  RF+Quantile beats {name:14s} on {wins:3d}/{len(seeds)} seeds; "
              f"mean MAE difference {diff.mean():8.3f}pp (sd {diff.std():.3f})")

    print(f"\nSaved per-seed results to {RESULTS_CSV}")
    print(f"Saved summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
