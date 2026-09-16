"""Reviewer Major Issue 9 (S2.1) verification, at the level the reviewer
actually asked for: not just "are the features scale-invariant?" but
"rescale every instance by 0.01, 10, and 1000, recompute the features, and
confirm heuristic and gap predictions are unchanged."

This fits ONE Stage 1 classifier + ONE Stage 2 gap regressor (RF+Quantile,
SEED=1, the paper's chosen pipeline), then for a sample of real instances
recomputes the full feature vector at each scale factor and pushes it
through both stages, comparing the resulting predictions against the
unscaled baseline.

Usage:
    python experiments/verify_scale_invariance_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_dataset import classify_edge_weight_type  # noqa: E402
from src.config import DATASET_PATH  # noqa: E402
from src.features import STAGE1_INSTANCE_FEATURES, extract_all_features  # noqa: E402
from src.models import (  # noqa: E402
    fit_gap_regressor_rf_quantile,
    fit_random_forest,
    prepare_stage2_split,
)
from src.tsp_io import (  # noqa: E402
    get_distance_matrix,
    load_raw_euclidean_2d,
    load_raw_explicit,
    load_raw_geo,
)

SCALE_FACTORS = (0.01, 10, 1000)

# A spread of instance types/sizes, including brg180 (the manuscript's own
# example, whose avg_violation_magnitude the reviewer singled out).
SAMPLE_INSTANCES = ["burma14", "gr17", "att48", "eil51", "brg180", "gr96", "kroA100"]


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    split = prepare_stage2_split(df)
    pipeline_class = fit_random_forest(
        split["best_rows_train"][STAGE1_INSTANCE_FEATURES],
        split["best_rows_train"]["heuristic"],
    )
    predict_gap = fit_gap_regressor_rf_quantile(split["df_gap"], split["train_files"])["predict"]

    instances = {}
    for loader in (load_raw_euclidean_2d, load_raw_explicit, load_raw_geo):
        instances.update(loader())

    rows = []
    for name in SAMPLE_INSTANCES:
        if name not in instances:
            print(f"  (skipping {name}: not found on disk)")
            continue
        inst = instances[name]
        base_matrix = np.asarray(get_distance_matrix(inst), dtype=float)
        inst_type = classify_edge_weight_type(inst.get("edge_weight_type"))

        def predict_at(matrix):
            feats = extract_all_features(matrix)
            feats["type"] = inst_type
            X = pd.DataFrame([feats])[STAGE1_INSTANCE_FEATURES]
            h = pipeline_class.predict(X)[0]
            point, lower, upper = predict_gap(feats, h)
            return feats, h, point

        base_feats, base_h, base_gap = predict_at(base_matrix)

        for k in SCALE_FACTORS:
            feats, h, gap = predict_at(base_matrix * k)
            rel_gap_diff = abs(gap - base_gap) / max(abs(base_gap), 1e-12)
            rows.append(
                {
                    "instance": name,
                    "n": inst["dimension"],
                    "scale": k,
                    "base_heuristic": base_h,
                    "scaled_heuristic": h,
                    "heuristic_same": h == base_h,
                    "base_gap": base_gap,
                    "scaled_gap": gap,
                    "rel_gap_diff": rel_gap_diff,
                    "gap_same": rel_gap_diff <= 1e-6,
                    "base_std_edge_weight": base_feats["std_edge_weight"],
                    "scaled_std_edge_weight": feats["std_edge_weight"],
                }
            )

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n=== Per-instance prediction stability under rescaling ===")
    print(
        res[
            [
                "instance", "n", "scale", "base_heuristic", "scaled_heuristic",
                "heuristic_same", "base_gap", "scaled_gap", "rel_gap_diff", "gap_same",
            ]
        ].to_string(index=False)
    )

    n_tests = len(res)
    n_h_same = int(res["heuristic_same"].sum())
    n_g_same = int(res["gap_same"].sum())
    print(f"\nHeuristic prediction unchanged: {n_h_same}/{n_tests}")
    print(f"Gap prediction unchanged (rel diff <= 1e-6): {n_g_same}/{n_tests}")

    print("\n=== Why: std_edge_weight is still a raw-units model input ===")
    print(
        res[["instance", "scale", "base_std_edge_weight", "scaled_std_edge_weight"]]
        .to_string(index=False)
    )

    out = Path(__file__).resolve().parent.parent / "results" / "scale_invariance_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
