"""R3 + m4 — feature diagnostics: collinearity (correlation, VIF) alongside
importance from every model that consumes these features.

Produces results/feature_diagnostics.csv with one row per numeric Stage 1
feature and columns:
    feature, rf_gini, xgb_gini, rf_permutation, stage2_gini, vif,
    strongest_correlate, strongest_r

Also writes results/feature_correlation_matrix.csv (the full 9x9 Pearson
matrix) for reference; the paper reports only the pairs exceeding |r| = 0.7,
because a 9x9 grid is unreadable at the paper's column width.

VIF is computed as 1/(1 - R^2) from regressing each standardized feature on
all the others (statsmodels is not installed in this environment; sklearn's
LinearRegression gives the identical quantity).

All importances are read from the existing model functions at seed=1. No
model is retrained with different settings and no hyperparameter or seed is
changed.

Run from repo root: python -m experiments.feature_diagnostics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import STAGE1_NUMERIC_FEATURES
from src.models import (
    SEED,
    build_random_forest_pipeline,
    build_xgboost_tuned_pipeline,
    evaluate_stage1_cv,
    prepare_stage1_data,
    run_integrated_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "Dataset.csv"
RESULTS_DIR = REPO_ROOT / "results"

CORR_THRESHOLD = 0.7

# Values independently computed beforehand; the script verifies against these
# and STOPS on disagreement rather than silently adopting different numbers.
EXPECTED_VIF = {
    # Recomputed after dropping std_edge_weight (S2.1 fix, see
    # experiments/drop_std_edge_weight.py). Removing one correlated feature
    # slightly reduces VIF pressure on the rest; pairwise correlations among
    # the remaining features are unaffected (verified below).
    "triangle_violation_rate": 84.12,
    "avg_violation_magnitude": 36.23,
    "cv_edge_weight": 25.20,
    "mst_weight": 5.66,
}
EXPECTED_CORR_PAIRS = {
    ("avg_violation_magnitude", "triangle_violation_rate"): 0.914,
    ("cv_edge_weight", "triangle_violation_rate"): 0.846,
    ("min_edge_weight", "mst_weight"): 0.780,
}


def compute_vif(X: pd.DataFrame) -> dict[str, float]:
    Xs = ((X - X.mean()) / X.std()).values
    vifs = {}
    for i, col in enumerate(X.columns):
        y = Xs[:, i]
        Z = np.delete(Xs, i, axis=1)
        r2 = LinearRegression().fit(Z, y).score(Z, y)
        vifs[col] = float(1.0 / (1.0 - r2)) if r2 < 1.0 else float("inf")
    return vifs


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    _, best_rows = prepare_stage1_data(df)
    X = best_rows[STAGE1_NUMERIC_FEATURES]

    # --- collinearity -----------------------------------------------------
    corr = X.corr()
    corr.to_csv(RESULTS_DIR / "feature_correlation_matrix.csv")
    vifs = compute_vif(X)

    strong_pairs = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) > CORR_THRESHOLD:
                strong_pairs.append((corr.index[i], corr.columns[j], float(r)))

    # strongest correlate per feature (excluding self)
    strongest = {}
    for feat in STAGE1_NUMERIC_FEATURES:
        others = corr[feat].drop(index=feat)
        partner = others.abs().idxmax()
        strongest[feat] = (partner, float(others[partner]))

    # --- importances ------------------------------------------------------
    rf = evaluate_stage1_cv(df, build_random_forest_pipeline, seed=SEED)
    xgb = evaluate_stage1_cv(
        df, build_xgboost_tuned_pipeline, seed=SEED, needs_label_encoding=True
    )
    stage2 = run_integrated_pipeline(df, backbone="rf", uncertainty_method="quantile", seed=SEED)

    def as_map(imp_df: pd.DataFrame) -> dict[str, float]:
        return dict(zip(imp_df["feature"], imp_df["importance"]))

    rf_gini = as_map(rf["gini_importance"])
    xgb_gini = as_map(xgb["gini_importance"])
    rf_perm = as_map(rf["permutation_importance"])
    s2_gini = as_map(stage2["gap_feature_importance"])

    rows = []
    for feat in STAGE1_NUMERIC_FEATURES:
        partner, r = strongest[feat]
        rows.append(
            {
                "feature": feat,
                "rf_gini": round(rf_gini.get(feat, float("nan")), 4),
                "xgb_gini": round(xgb_gini.get(feat, float("nan")), 4),
                "rf_permutation": round(rf_perm.get(feat, float("nan")), 4),
                "stage2_gini": round(s2_gini.get(feat, float("nan")), 4),
                "vif": round(vifs[feat], 2),
                "strongest_correlate": partner,
                "strongest_r": round(r, 3),
            }
        )
    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "feature_diagnostics.csv"
    out.to_csv(out_path, index=False)

    # --- report + verification -------------------------------------------
    print("=" * 78)
    print("FEATURE DIAGNOSTICS")
    print("=" * 78)
    print(out.to_string(index=False))
    print()
    print(f"Pairs with |r| > {CORR_THRESHOLD}: {len(strong_pairs)}")
    for a, b, r in strong_pairs:
        print(f"   {a} <-> {b}: r = {r:.3f}")
    print()

    ok = True
    print("VERIFICATION vs. independently computed values:")
    for feat, expected in EXPECTED_VIF.items():
        got = round(vifs[feat], 2)
        match = abs(got - expected) < 0.05
        ok = ok and match
        print(f"   VIF {feat:26s} expected {expected:7.2f}  got {got:7.2f}  {'OK' if match else 'MISMATCH'}")
    got_pairs = {tuple(sorted((a, b))): round(r, 3) for a, b, r in strong_pairs}
    exp_pairs = {tuple(sorted(k)): v for k, v in EXPECTED_CORR_PAIRS.items()}
    if set(got_pairs) != set(exp_pairs):
        ok = False
        print(f"   MISMATCH: correlation pair set differs.")
        print(f"     expected {sorted(exp_pairs)}")
        print(f"     got      {sorted(got_pairs)}")
    else:
        for k, expected in exp_pairs.items():
            got = got_pairs[k]
            match = abs(abs(got) - abs(expected)) < 0.002
            ok = ok and match
            print(f"   r {k[0]}/{k[1]}: expected {expected:.3f} got {got:.3f}  {'OK' if match else 'MISMATCH'}")

    print()
    if not ok:
        print("VERIFICATION FAILED — stopping. Values differ from those independently computed.")
        sys.exit(1)
    print("VERIFICATION PASSED.")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
