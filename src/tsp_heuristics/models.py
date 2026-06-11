"""Load bundled pre-trained models and run heuristic + gap prediction."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .features import NUMERIC_FEATURE_NAMES

_HEURISTICS = ["CH", "Greedy", "Insertion", "NN"]

# Map model_type → (clf_filename_template, gap_filename_template)
_MODEL_FILES: dict[str, tuple[str, str]] = {
    "rf":  ("rf_s{seed}_clf.pkl.gz",  "rf_s{seed}_qrf.pkl.gz"),
    "xgb": ("xgb_s{seed}_clf.pkl.gz", "xgb_s{seed}_qreg.pkl.gz"),
    "lr":  ("lr_clf.pkl.gz",           "lr_reg.pkl.gz"),
}

# Cache loaded models in memory (key: filename)
_cache: dict[str, Any] = {}


def _model_path(filename: str) -> Path:
    data_pkg = files("tsp_heuristics.data") / "models" / filename
    # importlib.resources path → real Path
    return Path(str(data_pkg))


def _load(filename: str) -> Any:
    if filename not in _cache:
        path = _model_path(filename)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run `python scripts/train_and_export.py` to generate it."
            )
        _cache[filename] = joblib.load(path)
    return _cache[filename]


def predict(
    numeric_features: np.ndarray,
    edge_type: str,
    model_type: str = "rf",
    seed: int = 1,
    intervals: bool = True,
) -> dict[str, Any]:
    """Predict best heuristic and optimality gap.

    Parameters
    ----------
    numeric_features:
        Shape-(11,) array from ``features.extract_features``.
        Order: n, min_edge_weight, max_edge_weight, std_edge_weight,
               cv_edge_weight, edge_weight_skewness, pct_short_edges,
               triangle_violation_rate, avg_violation_magnitude,
               mst_weight, nn_cost_over_mst
    edge_type:
        One of 'EUC_2D', 'Geo', 'Explicit' (from parsers.load_distance_matrix).
    model_type:
        'rf' | 'xgb' | 'lr'
    seed:
        1, 2, or 3  (ignored for 'lr')
    intervals:
        If True and model supports it, return 95% CI bounds.

    Returns
    -------
    dict with keys:
      heuristic         str    — predicted best heuristic
      heuristic_proba   dict   — {heuristic: probability}
      gap_pct           float  — predicted optimality gap (%)
      gap_lower_95      float | None
      gap_upper_95      float | None
    """
    if model_type not in _MODEL_FILES:
        raise ValueError(f"model_type must be one of {list(_MODEL_FILES)}, got {model_type!r}")
    if seed not in (1, 2, 3):
        raise ValueError(f"seed must be 1, 2, or 3, got {seed!r}")

    clf_tpl, gap_tpl = _MODEL_FILES[model_type]
    clf_file = clf_tpl.format(seed=seed)
    gap_file = gap_tpl.format(seed=seed)

    clf_data = _load(clf_file)
    clf_pipeline = clf_data["pipeline"]
    label_encoder = clf_data["label_encoder"]

    # Build single-row DataFrame for the classifier
    X_clf = _make_instance_df(numeric_features, edge_type)

    # Predict heuristic
    y_enc = clf_pipeline.predict(X_clf)[0]
    heuristic = label_encoder.inverse_transform([y_enc])[0]

    # Probabilities (if available)
    proba_dict: dict[str, float]
    if hasattr(clf_pipeline, "predict_proba"):
        proba = clf_pipeline.predict_proba(X_clf)[0]
        classes = label_encoder.inverse_transform(clf_pipeline.classes_)
        proba_dict = {str(c): float(p) for c, p in zip(classes, proba)}
    else:
        proba_dict = {h: (1.0 if h == heuristic else 0.0) for h in _HEURISTICS}

    # Gap prediction
    gap_data = _load(gap_file)
    gap_prep = gap_data["preprocessor"]

    X_gap = _make_gap_df(numeric_features, edge_type, heuristic)

    if model_type == "rf":
        qrf = gap_data["regressor"]
        X_proc = gap_prep.transform(X_gap)
        if intervals:
            preds = qrf.predict(X_proc, quantiles=[0.05, 0.5, 0.95])
            gap_lower = float(np.expm1(preds[0, 0]))
            gap_med = float(np.expm1(preds[0, 1]))
            gap_upper = float(np.expm1(preds[0, 2]))
        else:
            gap_med = float(np.expm1(qrf.predict(X_proc, quantiles=0.5)[0]))
            gap_lower = gap_upper = None

    elif model_type == "xgb":
        qmodels = gap_data["models"]
        X_proc = gap_prep.transform(X_gap)
        gap_med = float(np.expm1(qmodels["q50"].predict(X_proc)[0]))
        if intervals:
            gap_lower = float(np.expm1(qmodels["q05"].predict(X_proc)[0]))
            gap_upper = float(np.expm1(qmodels["q95"].predict(X_proc)[0]))
        else:
            gap_lower = gap_upper = None

    else:  # lr — no quantile support
        linreg = gap_data["regressor"]
        X_proc = gap_prep.transform(X_gap)
        gap_med = float(np.expm1(linreg.predict(X_proc)[0]))
        gap_lower = gap_upper = None

    return {
        "heuristic": str(heuristic),
        "heuristic_proba": proba_dict,
        "gap_pct": round(gap_med, 2),
        "gap_lower_95": round(gap_lower, 2) if gap_lower is not None else None,
        "gap_upper_95": round(gap_upper, 2) if gap_upper is not None else None,
    }


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _make_instance_df(numeric_features: np.ndarray, edge_type: str) -> pd.DataFrame:
    row = {name: float(numeric_features[i]) for i, name in enumerate(NUMERIC_FEATURE_NAMES)}
    row["type"] = edge_type
    return pd.DataFrame([row])


def _make_gap_df(
    numeric_features: np.ndarray, edge_type: str, heuristic: str
) -> pd.DataFrame:
    row = {name: float(numeric_features[i]) for i, name in enumerate(NUMERIC_FEATURE_NAMES)}
    row["type"] = edge_type
    row["heuristic"] = heuristic
    return pd.DataFrame([row])
