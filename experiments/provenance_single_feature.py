"""MI-11 support: how much provenance signal does a single numeric feature carry?

The XGBoost section argues that Gini importance cannot settle whether the
classifier leans on provenance, because several numeric features redundantly
encode it. This script quantifies that: it fits a depth-2 decision tree to
predict is_Explicit from ONE feature at a time, under the same stratified
5-fold protocol (shuffle=True, random_state=1) used everywhere else.

Run from repo root: python -m experiments.provenance_single_feature
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_PATH  # noqa: E402
from src.models import prepare_stage1_data  # noqa: E402

FEATURES = ["cv_edge_weight", "min_edge_weight", "avg_violation_magnitude"]
RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    _, best = prepare_stage1_data(df)
    y = (best["type"] == "Explicit").astype(int)
    baseline = max(y.mean(), 1 - y.mean())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    rows = [{"feature": "(majority baseline)", "max_depth": "", "cv_accuracy": round(baseline, 4)}]
    print(f"is_Explicit majority baseline: {baseline*100:.1f}%")
    for f in FEATURES:
        acc = cross_val_score(
            DecisionTreeClassifier(max_depth=2, random_state=1), best[[f]], y, cv=cv
        ).mean()
        rows.append({"feature": f, "max_depth": 2, "cv_accuracy": round(acc, 4)})
        print(f"  {f:<26} depth-2 tree, single feature: {acc*100:.1f}%")

    out = RESULTS / "provenance_single_feature.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
