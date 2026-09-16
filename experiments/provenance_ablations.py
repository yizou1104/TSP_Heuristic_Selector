"""S2.4 — provenance / generalization ablations, plus the std_edge_weight decision.

Four independent experiments:

  A. TYPE ABLATION            drop the categorical `type` feature entirely and
                              re-run Stage 1 CV. If accuracy collapses, the
                              selector was leaning on provenance.

  B. TSPLIB-ONLY EVALUATION   two variants:
                              B1  standard CV, accuracy restricted to the 105
                                  TSPLIB instances (generated ones still in train)
                              B2  train on generated only, test on all TSPLIB
                              Both compared against the correct baseline for
                              that subset, not the whole-dataset baseline.

  C. LEAVE-ONE-FAMILY-OUT     hold out an entire synthetic family, train on
                              everything else, test on the held-out family.

  D. STD_EDGE_WEIGHT DROP     re-run Stage 1 CV and the full Seed-1 Stage 2
                              pipeline without `std_edge_weight`, to decide
                              whether keeping a non-scale-invariant feature is
                              justified by any measurable gain.

Run from repo root: python -m experiments.provenance_ablations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (
    STAGE1_CATEGORICAL_FEATURES,
    STAGE1_INSTANCE_FEATURES,
    STAGE1_NUMERIC_FEATURES,
)
from src.models import SEED, prepare_stage1_data

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def family_of(fname: str) -> str:
    if fname.startswith("family_A"):
        return "gen_A_random"
    if fname.startswith("family_B"):
        return "gen_B_clustered"
    if fname.startswith("christofides_hard"):
        return "gen_C_chard"
    return "TSPLIB"


def build_rf(numeric, categorical, seed=SEED):
    """Random Forest with the paper's published hyperparameters, over an
    arbitrary feature subset."""
    transformers = [("num", "passthrough", numeric)]
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    pre = ColumnTransformer(transformers)
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=-1
    )
    return Pipeline([("preprocess", pre), ("classifier", clf)])


def majority_baseline(y) -> tuple[str, float]:
    vc = pd.Series(y).value_counts()
    return vc.index[0], float(vc.iloc[0] / len(y))


def cv_accuracy(X, y, numeric, categorical, seed=SEED):
    pipe = build_rf(numeric, categorical, seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(pipe, X, y, cv=cv)
    return float(accuracy_score(y, pred)), pred


def main() -> None:
    df = pd.read_csv(REPO / "data" / "Dataset.csv")
    _, best = prepare_stage1_data(df)
    best = best.copy()
    best["family"] = best["file"].map(family_of)

    X_full = best[STAGE1_INSTANCE_FEATURES]
    y = best["heuristic"].values
    rows = []

    # ================= A. TYPE ABLATION =================
    print("=" * 78)
    print("A.  TYPE ABLATION — does the selector need the `type` label?")
    print("=" * 78)
    acc_with, _ = cv_accuracy(X_full, y, STAGE1_NUMERIC_FEATURES, STAGE1_CATEGORICAL_FEATURES)
    acc_without, _ = cv_accuracy(best[STAGE1_NUMERIC_FEATURES], y, STAGE1_NUMERIC_FEATURES, [])
    lbl, base = majority_baseline(y)
    print(f"  with    `type`: {acc_with*100:6.2f}%")
    print(f"  without `type`: {acc_without*100:6.2f}%")
    print(f"  change        : {(acc_without-acc_with)*100:+6.2f} pp")
    print(f"  (majority-class baseline, always '{lbl}': {base*100:.2f}%)")
    rows += [
        dict(experiment="A_type_ablation", condition="with_type", n_test=len(y), accuracy=acc_with),
        dict(experiment="A_type_ablation", condition="without_type", n_test=len(y), accuracy=acc_without),
    ]

    # ================= B. TSPLIB-ONLY =================
    print()
    print("=" * 78)
    print("B.  TSPLIB-ONLY EVALUATION")
    print("=" * 78)
    is_tsplib = (best["family"] == "TSPLIB").values
    y_tsplib = y[is_tsplib]
    lbl_t, base_t = majority_baseline(y_tsplib)
    print(f"  TSPLIB subset: {is_tsplib.sum()} instances")
    print(f"  baseline on this subset (always '{lbl_t}'): {base_t*100:.2f}%   <-- the number to beat")
    print()

    # B1: standard CV, accuracy measured only on TSPLIB instances
    _, pred_all = cv_accuracy(X_full, y, STAGE1_NUMERIC_FEATURES, STAGE1_CATEGORICAL_FEATURES)
    acc_b1 = float(accuracy_score(y_tsplib, pred_all[is_tsplib]))
    print(f"  B1  standard CV, scored on TSPLIB only : {acc_b1*100:6.2f}%  "
          f"({'+' if acc_b1>base_t else ''}{(acc_b1-base_t)*100:.2f} pp vs baseline)")

    # B2: train on generated ONLY, test on all TSPLIB
    gen = ~is_tsplib
    pipe = build_rf(STAGE1_NUMERIC_FEATURES, STAGE1_CATEGORICAL_FEATURES)
    pipe.fit(X_full[gen], y[gen])
    acc_b2 = float(accuracy_score(y_tsplib, pipe.predict(X_full[is_tsplib])))
    print(f"  B2  train on {gen.sum()} generated -> test on {is_tsplib.sum()} TSPLIB: {acc_b2*100:6.2f}%  "
          f"({'+' if acc_b2>base_t else ''}{(acc_b2-base_t)*100:.2f} pp vs baseline)")
    print(f"      training-set heuristic mix: {dict(pd.Series(y[gen]).value_counts())}")
    rows += [
        dict(experiment="B1_tsplib_scored", condition="cv_scored_on_tsplib", n_test=int(is_tsplib.sum()), accuracy=acc_b1),
        dict(experiment="B2_train_generated", condition="train_gen_test_tsplib", n_test=int(is_tsplib.sum()), accuracy=acc_b2),
        dict(experiment="B_baseline", condition=f"always_{lbl_t}", n_test=int(is_tsplib.sum()), accuracy=base_t),
    ]

    # ================= C. LEAVE-ONE-FAMILY-OUT =================
    print()
    print("=" * 78)
    print("C.  LEAVE-ONE-FAMILY-OUT")
    print("=" * 78)
    for fam in ["gen_A_random", "gen_B_clustered", "gen_C_chard", "TSPLIB"]:
        held = (best["family"] == fam).values
        if held.sum() == 0:
            continue
        pipe = build_rf(STAGE1_NUMERIC_FEATURES, STAGE1_CATEGORICAL_FEATURES)
        pipe.fit(X_full[~held], y[~held])
        acc = float(accuracy_score(y[held], pipe.predict(X_full[held])))
        lbl_h, base_h = majority_baseline(y[held])
        train_classes = sorted(set(y[~held]))
        held_classes = dict(pd.Series(y[held]).value_counts())
        print(f"  hold out {fam:16s} ({held.sum():3d} inst): acc {acc*100:6.2f}%  "
              f"| baseline (always '{lbl_h}') {base_h*100:6.2f}%")
        print(f"      held-out mix {held_classes}")
        print(f"      classes present in training: {train_classes}")
        rows.append(dict(experiment="C_leave_one_family_out", condition=f"held_{fam}",
                         n_test=int(held.sum()), accuracy=acc))
        rows.append(dict(experiment="C_baseline", condition=f"held_{fam}_majority",
                         n_test=int(held.sum()), accuracy=base_h))

    # ================= D. DROP std_edge_weight =================
    print()
    print("=" * 78)
    print("D.  DROPPING std_edge_weight  (the non-scale-invariant feature)")
    print("=" * 78)
    num_wo = [f for f in STAGE1_NUMERIC_FEATURES if f != "std_edge_weight"]
    acc_drop, _ = cv_accuracy(best[num_wo + STAGE1_CATEGORICAL_FEATURES], y, num_wo, STAGE1_CATEGORICAL_FEATURES)
    print(f"  Stage 1 CV with    std_edge_weight: {acc_with*100:6.2f}%")
    print(f"  Stage 1 CV without std_edge_weight: {acc_drop*100:6.2f}%")
    print(f"  change                            : {(acc_drop-acc_with)*100:+6.2f} pp")
    rows += [
        dict(experiment="D_drop_std_edge_weight", condition="with_std", n_test=len(y), accuracy=acc_with),
        dict(experiment="D_drop_std_edge_weight", condition="without_std", n_test=len(y), accuracy=acc_drop),
    ]

    out = RESULTS / "provenance_ablations.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
