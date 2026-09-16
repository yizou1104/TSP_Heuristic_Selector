"""Full-dataset extension of verify_scale_invariance_predictions.py.

The original script (Reviewer Major Issue 9 / S2.1) tested a hand-picked
spread of 7 instances. The reviewer's literal request was "rescale every
instance by 0.01, 10, and 1000" -- this script attempts that over every
instance actually listed in data/Dataset.csv (the authoritative 160-instance
set), using the identical pipeline (RF classifier + RF+Quantile gap
regressor, SEED=1) and identical logic as the original script.

Source files are located as follows:
  - EUC_2D / Explicit-original / Geo: data/raw/{euclidean_2d,explicit,geo}/
  - Explicit generated (family_A_random, family_B_clustered): both live
    together under data/generated/clustered/ (a pre-existing directory
    layout quirk -- data/generated/random/ is empty).
  - Explicit generated (christofides_hard): data/generated/christofides_hard/

Any of the 160 instances whose source file cannot be found on disk is
reported by name and excluded, rather than silently skipped or faked.

Usage:
    python experiments/verify_scale_invariance_full.py
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
    load_generated,
)

SCALE_FACTORS = (0.01, 10, 1000)


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    official_files = list(df["file"].unique())
    print(f"Dataset.csv lists {len(official_files)} official instances.")

    split = prepare_stage2_split(df)
    pipeline_class = fit_random_forest(
        split["best_rows_train"][STAGE1_INSTANCE_FEATURES],
        split["best_rows_train"]["heuristic"],
    )
    predict_gap = fit_gap_regressor_rf_quantile(split["df_gap"], split["train_files"])["predict"]

    instances = {}
    for loader in (load_raw_euclidean_2d, load_raw_explicit, load_raw_geo):
        instances.update(loader())
    # Generated Explicit families: both random and clustered live under the
    # same 'clustered' output directory on disk.
    for fam in ("random", "clustered"):
        try:
            instances.update(load_generated("clustered"))
            break
        except Exception:
            pass
    instances.update(load_generated("christofides_hard"))

    found = [f for f in official_files if f in instances]
    missing = [f for f in official_files if f not in instances]
    print(f"Found on disk: {len(found)}/{len(official_files)}")
    if missing:
        print(f"MISSING (excluded, source file not present locally): {missing}")

    rows = []
    for i, name in enumerate(found):
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
                    "type": inst_type,
                    "scale": k,
                    "base_heuristic": base_h,
                    "scaled_heuristic": h,
                    "heuristic_same": h == base_h,
                    "base_gap": base_gap,
                    "scaled_gap": gap,
                    "rel_gap_diff": rel_gap_diff,
                    "gap_same": rel_gap_diff <= 1e-6,
                }
            )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(found)} instances done...")

    res = pd.DataFrame(rows)

    n_tests = len(res)
    n_h_same = int(res["heuristic_same"].sum())
    n_g_same = int(res["gap_same"].sum())
    print(f"\n=== Full sweep: {len(found)} of {len(official_files)} official instances x 3 scales = {n_tests} tests ===")
    print(f"Heuristic prediction unchanged: {n_h_same}/{n_tests}")
    print(f"Gap prediction unchanged (rel diff <= 1e-6): {n_g_same}/{n_tests}")

    mismatches = res[~(res["heuristic_same"] & res["gap_same"])]
    print(f"\n=== All {len(mismatches)} mismatching tests, by scale ===")
    print(mismatches["scale"].value_counts().to_string())

    print("\n=== Heuristic-flip mismatches (the more serious kind) ===")
    hflips = res[~res["heuristic_same"]]
    pd.set_option("display.width", 250)
    print(hflips[["instance", "n", "type", "scale", "base_heuristic", "scaled_heuristic"]].to_string(index=False))

    print("\n=== Gap-only mismatch magnitude distribution (rel_gap_diff, among mismatches) ===")
    print(mismatches[~mismatches["heuristic_same"] == False]["rel_gap_diff"].describe().to_string())
    print(mismatches["rel_gap_diff"].describe().to_string())

    out = Path(__file__).resolve().parent.parent / "results" / "scale_invariance_predictions_full.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
