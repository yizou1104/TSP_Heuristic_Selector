"""Regenerate the Stage 1 (heuristic selector) numbers the manuscript quotes,
on the CURRENT post-S2.1 feature set, plus the full Stage 2 RF+Quantile gap
feature-importance table. Needed because renormalizing min_edge_weight and
avg_violation_magnitude changes the model inputs, so the previously reported
CV accuracies, regret figures, and feature importances are stale.

Usage:
    python experiments/refresh_stage1_numbers.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.models as _models  # noqa: E402

# evaluate_stage1_cv calls gini_feature_importance unconditionally, but
# LogisticRegression exposes coef_ rather than feature_importances_. Degrade
# to None for that model instead of crashing (read-only shim; src/models.py
# is left untouched).
_orig_gini = _models.gini_feature_importance


def _safe_gini(pipeline):
    try:
        return _orig_gini(pipeline)
    except AttributeError:
        return None


_models.gini_feature_importance = _safe_gini

from src.config import DATASET_PATH  # noqa: E402
from src.models import (  # noqa: E402
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    build_xgboost_tuned_pipeline,
    evaluate_stage1_cv,
    fit_gap_regressor_rf_quantile,
    prepare_stage2_split,
)

MODELS = [
    ("Logistic Regression", build_logistic_regression_pipeline, False),
    ("Random Forest", build_random_forest_pipeline, False),
    ("XGBoost", build_xgboost_tuned_pipeline, True),
]


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    pd.set_option("display.width", 250)

    for name, builder, needs_le in MODELS:
        r = evaluate_stage1_cv(df, builder, needs_label_encoding=needs_le)
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        print(f"  fold_accuracy       : {[round(float(a), 4) for a in r['fold_accuracy']]}")
        print(f"  mean_accuracy       : {r['mean_accuracy']:.4f}")
        print(f"  std_accuracy        : {r['std_accuracy']:.4f}")
        print(f"  train_accuracy_mean : {r['train_accuracy_mean']:.4f}")
        print(f"  exact_match_rate    : {r['exact_match_rate']:.4f}")
        print(f"  mean_regret         : {r['mean_regret']:.4f}")
        print(f"  median_regret       : {r['median_regret']:.4f}")
        print(f"  catastrophic_rate   : {r['catastrophic_failure_rate']:.4f}")

        # conditional regret among mistakes (reviewer Major Issue 6)
        oof = r["oof_predictions"].dropna(subset=["regret"])
        wrong = oof[oof["true_heuristic"] != oof["predicted_heuristic"]]
        if len(wrong):
            print(f"  n_mistakes          : {len(wrong)} of {len(oof)}")
            print(f"  mean regret | wrong : {wrong['regret'].mean():.4f}")
            print(f"  max regret  | wrong : {wrong['regret'].max():.4f}")
            worst = wrong.nlargest(3, "regret")
            print("  worst mistakes:")
            print(worst[["file", "true_heuristic", "predicted_heuristic", "regret"]]
                  .to_string(index=False))

        if r["gini_importance"] is not None:
            print("\n  Gini importance (top 6):")
            print(r["gini_importance"].head(6).to_string(index=False))
        print("\n  Permutation importance (top 6):")
        print(r["permutation_importance"].head(6).to_string(index=False))
        print("\n  Classification report:")
        print(r["classification_report"])

    # Full Stage 2 RF+Quantile gap-regressor importance (paper quotes shares)
    split = prepare_stage2_split(df)
    gap_fit = fit_gap_regressor_rf_quantile(split["df_gap"], split["train_files"])
    imp = gap_fit["feature_importance"]
    print("\n" + "=" * 70)
    print("Stage 2 RF+Quantile gap-regressor feature importance (FULL)")
    print("=" * 70)
    print(imp.to_string(index=False))
    heur = imp[imp["feature"].str.startswith("heuristic_")]
    print(f"\n  heuristic_* indicators total share: {heur['importance'].sum():.4f}")
    top2 = imp.head(2)
    print(f"  top-2 ({', '.join(top2['feature'])}) total share: {top2['importance'].sum():.4f}")


if __name__ == "__main__":
    main()
