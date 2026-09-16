"""Decide empirically whether `std_edge_weight` should be dropped.

It is the one feature that is not scale invariant, which is why Stage 2
predictions reproduce in only 9 of 21 rescaling tests. The paper currently
keeps it and discloses the gap. This script tests whether keeping it buys
anything measurable:

  1. Stage 2 (Seed 1) metrics for all four pipelines, with and without it.
  2. Scale-invariance of Stage 2 predictions without it — does 9/21 become 21/21?

The feature lists are patched in memory only; nothing on disk changes.

Run from repo root: python -m experiments.drop_std_edge_weight
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.features as F
import src.models as M

REPO = Path(__file__).resolve().parents[1]
PIPELINES = [("rf", "quantile"), ("rf", "conformal"), ("xgb", "quantile"), ("xgb", "conformal")]
SCALES = [0.01, 10.0, 1000.0]


def run_all(df: pd.DataFrame) -> dict:
    out = {}
    for backbone, method in PIPELINES:
        r = M.run_integrated_pipeline(df, backbone=backbone, uncertainty_method=method, seed=M.SEED)
        out[f"{backbone}+{method}"] = dict(
            mae=r["gap_mae"], r2=r["r2_log"], cov=r["coverage"], acc=r["classifier_accuracy"]
        )
    return out


def patch_features(drop: str | None):
    """Rebuild the module-level feature lists with `drop` removed."""
    base_num = ["n", "min_edge_weight", "std_edge_weight", "cv_edge_weight",
                "edge_weight_skewness", "pct_short_edges", "triangle_violation_rate",
                "avg_violation_magnitude", "mst_weight"]
    num = [f for f in base_num if f != drop]
    inst = num + ["type"]
    for mod in (F, M):
        mod.STAGE1_NUMERIC_FEATURES = num
        mod.STAGE1_INSTANCE_FEATURES = inst
        mod.STAGE2_FEATURE_COLUMNS = inst + ["heuristic"]
    F.STAGE1_NUMERIC_FEATURES = num
    F.STAGE1_INSTANCE_FEATURES = inst
    F.STAGE2_FEATURE_COLUMNS = inst + ["heuristic"]


def scale_invariance_stage2(df: pd.DataFrame, label: str) -> tuple[int, int]:
    """Rescale a sample of instances and count how many Stage 2 gap predictions
    are exactly reproduced. Mirrors the S2.1 verification design."""
    split = M.prepare_stage2_split(df, seed=M.SEED)
    fit = M.fit_gap_regressor_rf_quantile(split["df_gap"], split["train_files"], seed=M.SEED)
    predict = fit["predict"]

    best = split["best_rows"]
    sample = best[best["file"].isin(["burma14", "ulysses16", "gr21", "fri26", "bays29",
                                     "eil76", "brg180", "pr76"])]
    stable = total = 0
    for _, row in sample.iterrows():
        feats = {f: row[f] for f in F.STAGE1_INSTANCE_FEATURES}
        base_pt, _, _ = predict(feats, row["heuristic"])
        for k in SCALES:
            scaled = dict(feats)
            # dimensionful features scale with k; ratios and counts do not
            for f in ("min_edge_weight", "std_edge_weight", "avg_violation_magnitude"):
                if f in scaled:
                    # min_edge_weight and avg_violation_magnitude are already
                    # normalized ratios post-S2.1; only std_edge_weight is raw
                    if f == "std_edge_weight":
                        scaled[f] = scaled[f] * k
            pt, _, _ = predict(scaled, row["heuristic"])
            total += 1
            if np.isclose(pt, base_pt, rtol=1e-12, atol=1e-12):
                stable += 1
    print(f"  {label}: Stage 2 gap prediction reproduced in {stable}/{total} rescaling tests")
    return stable, total


def main() -> None:
    df = pd.read_csv(REPO / "data" / "Dataset.csv")

    print("=" * 78)
    print("STAGE 2 (Seed 1) — WITH std_edge_weight  [the published configuration]")
    print("=" * 78)
    patch_features(None)
    with_std = run_all(df)
    for k, v in with_std.items():
        print(f"  {k:16s} MAE {v['mae']:7.2f}pp   R2(log) {v['r2']:8.4f}   coverage {v['cov']*100:6.2f}%")
    inv_with = scale_invariance_stage2(df, "with std_edge_weight   ")

    print()
    print("=" * 78)
    print("STAGE 2 (Seed 1) — WITHOUT std_edge_weight")
    print("=" * 78)
    patch_features("std_edge_weight")
    without_std = run_all(df)
    for k, v in without_std.items():
        w = with_std[k]
        print(f"  {k:16s} MAE {v['mae']:7.2f}pp ({v['mae']-w['mae']:+6.2f})   "
              f"R2(log) {v['r2']:8.4f} ({v['r2']-w['r2']:+7.4f})   "
              f"coverage {v['cov']*100:6.2f}% ({(v['cov']-w['cov'])*100:+5.2f})")
    inv_without = scale_invariance_stage2(df, "without std_edge_weight")

    rows = []
    for k in with_std:
        rows.append(dict(pipeline=k, variant="with_std", **with_std[k]))
        rows.append(dict(pipeline=k, variant="without_std", **without_std[k]))
    out = REPO / "results" / "drop_std_edge_weight.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  scale invariance: {inv_with[0]}/{inv_with[1]} -> {inv_without[0]}/{inv_without[1]}")
    d_mae = np.mean([without_std[k]["mae"] - with_std[k]["mae"] for k in with_std])
    d_r2 = np.mean([without_std[k]["r2"] - with_std[k]["r2"] for k in with_std])
    print(f"  mean MAE change across 4 pipelines : {d_mae:+.3f} pp")
    print(f"  mean R2(log) change                : {d_r2:+.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
