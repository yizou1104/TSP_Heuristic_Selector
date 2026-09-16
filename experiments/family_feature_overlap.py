"""Is the leave-one-family-out failure an extrapolation problem (the held-out
family occupies a genuinely novel region of feature space, so no model could
bridge it) or a within-distribution generalization failure (the held-out
family overlaps existing feature space but the model still gets it wrong,
which would be more concerning)?

Two independent checks per family:
  1. Per-feature range overlap: what fraction of the held-out family's values
     fall inside the training data's [min, max] range for each feature.
  2. A plain 1-nearest-neighbor probe in standardized feature space: for each
     held-out instance, find its nearest training instance and check whether
     that neighbor's label matches. This is model-agnostic -- if even the
     single most similar training instance carries the wrong label, no
     classifier could get this instance right from structure alone.

Run from repo root: python -m experiments.family_feature_overlap
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import STAGE1_NUMERIC_FEATURES
from src.models import prepare_stage1_data

REPO = Path(__file__).resolve().parents[1]


def family_of(fname: str) -> str:
    if fname.startswith("family_A"):
        return "gen_A_random"
    if fname.startswith("family_B"):
        return "gen_B_clustered"
    if fname.startswith("christofides_hard"):
        return "gen_C_chard"
    return "TSPLIB"


def main() -> None:
    df = pd.read_csv(REPO / "data" / "Dataset.csv")
    _, best = prepare_stage1_data(df)
    best = best.copy()
    best["family"] = best["file"].map(family_of)

    X = best[STAGE1_NUMERIC_FEATURES]
    Xs = (X - X.mean()) / X.std()  # standardize once over the whole dataset
    y = best["heuristic"].values
    fam = best["family"].values

    rows = []
    for f in ["gen_A_random", "gen_B_clustered", "gen_C_chard", "TSPLIB"]:
        held = fam == f
        train = ~held
        print("=" * 78)
        print(f"FAMILY: {f}  ({held.sum()} instances)")
        print("=" * 78)

        # --- 1. per-feature range overlap ---
        print("  per-feature: % of held-out values inside training [min, max]")
        overlap_fracs = []
        for feat in STAGE1_NUMERIC_FEATURES:
            tr_min, tr_max = X.loc[train, feat].min(), X.loc[train, feat].max()
            held_vals = X.loc[held, feat]
            inside = ((held_vals >= tr_min) & (held_vals <= tr_max)).mean()
            overlap_fracs.append(inside)
            marker = "" if inside > 0.5 else "  <-- mostly OUTSIDE training range"
            print(f"    {feat:26s} train=[{tr_min:9.3f}, {tr_max:9.3f}]  "
                  f"held-out inside: {inside*100:5.1f}%{marker}")
        mean_overlap = float(np.mean(overlap_fracs))
        print(f"  mean feature-range overlap: {mean_overlap*100:.1f}%")

        # --- 2. 1-nearest-neighbor probe ---
        d = cdist(Xs.loc[held].values, Xs.loc[train].values, metric="euclidean")
        nn_idx = d.argmin(axis=1)
        nn_dist = d[np.arange(len(d)), nn_idx]
        nn_label = y[train][nn_idx]
        true_label = y[held]
        nn_acc = float((nn_label == true_label).mean())
        print(f"  1-NN probe: nearest-training-neighbor label matches true label "
              f"{int((nn_label==true_label).sum())}/{held.sum()} times ({nn_acc*100:.1f}%)")
        print(f"  mean distance to nearest training neighbor (standardized units): {nn_dist.mean():.2f}")
        # for context: typical within-dataset nearest-neighbor distance
        d_all = cdist(Xs.values, Xs.values, metric="euclidean")
        np.fill_diagonal(d_all, np.inf)
        typical_nn = np.median(d_all.min(axis=1))
        print(f"  (typical nearest-neighbor distance anywhere in the dataset: {typical_nn:.2f})")
        print()

        rows.append(dict(
            family=f, n=int(held.sum()), mean_feature_overlap=mean_overlap,
            nn_label_match_rate=nn_acc, mean_nn_distance=float(nn_dist.mean()),
            typical_dataset_nn_distance=float(typical_nn),
        ))

    out = REPO / "results" / "family_feature_overlap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
