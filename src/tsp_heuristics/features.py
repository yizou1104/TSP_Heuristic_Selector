"""Extract the 9 numeric structural features from a TSP distance matrix.

Feature order matches the training pipeline:
  n, min_edge_weight, std_edge_weight, cv_edge_weight,
  edge_weight_skewness, pct_short_edges, triangle_violation_rate,
  avg_violation_magnitude, mst_weight

Note: max_edge_weight and nn_cost_over_mst were removed from the feature
set after analysis showed they contributed to overfitting.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

NUMERIC_FEATURE_NAMES: list[str] = [
    "n",
    "min_edge_weight",
    "std_edge_weight",
    "cv_edge_weight",
    "edge_weight_skewness",
    "pct_short_edges",
    "triangle_violation_rate",
    "avg_violation_magnitude",
    "mst_weight",
]

# Full feature list including the categorical 'type' column used by the model.
ALL_FEATURE_NAMES: list[str] = NUMERIC_FEATURE_NAMES + ["type"]


def extract_features(
    dist: np.ndarray,
    n_triangle_samples: int = 200_000,
    seed: int = 42,
) -> np.ndarray:
    """Return shape-(9,) float64 array of numeric features.

    Parameters
    ----------
    dist:
        Symmetric n×n float64 distance matrix (diagonal must be 0).
    n_triangle_samples:
        Number of random triplets to sample for triangle inequality check.
    seed:
        RNG seed for reproducibility.
    """
    n = dist.shape[0]
    edges = _upper_tri(dist)

    min_w = float(np.min(edges[edges > 0])) if np.any(edges > 0) else 0.0
    mean_w = float(np.mean(edges))
    std_w = float(np.std(edges, ddof=0))
    cv_w = std_w / mean_w if mean_w > 0 else 0.0
    skew_w = _skewness(edges, mean_w, std_w)
    pct_short = _pct_short_edges(edges)
    tri_rate, tri_mag = _triangle_violations(dist, n, n_triangle_samples, seed)
    mst_w = _mst_weight(dist)

    return np.array([
        n, min_w, std_w, cv_w, skew_w,
        pct_short, tri_rate, tri_mag, mst_w,
    ], dtype=np.float64)


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _upper_tri(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    idx = np.triu_indices(n, k=1)
    return dist[idx]


def _skewness(edges: np.ndarray, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    z = (edges - mean) / std
    return float(np.mean(z ** 3))


def _pct_short_edges(edges: np.ndarray, percentile: float = 10.0) -> float:
    threshold = np.percentile(edges, percentile)
    return float(np.mean(edges <= threshold))


def _triangle_violations(
    dist: np.ndarray,
    n: int,
    n_samples: int,
    seed: int,
) -> tuple[float, float]:
    if n < 3:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    # Sample triplets as three independent indices; reject where any two are equal
    total_violations = 0
    total_magnitude = 0.0
    trials = 0

    batch = 10_000
    remaining = n_samples
    while remaining > 0:
        k = min(batch, remaining)
        idx = rng.integers(0, n, size=(k, 3))
        # Filter out degenerate triplets (any two equal)
        mask = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2])
        idx = idx[mask]
        if len(idx) == 0:
            remaining -= k
            continue
        i, j, k_ = idx[:, 0], idx[:, 1], idx[:, 2]
        dij = dist[i, j]
        djk = dist[j, k_]
        dik = dist[i, k_]
        violation = dik - (dij + djk)
        viol_mask = violation > 0
        total_violations += int(viol_mask.sum())
        total_magnitude += float(violation[viol_mask].sum())
        trials += len(idx)
        remaining -= k

    if trials == 0:
        return 0.0, 0.0
    rate = total_violations / trials
    mag = total_magnitude / total_violations if total_violations > 0 else 0.0
    return rate, mag


def _mst_weight(dist: np.ndarray) -> float:
    sparse = csr_matrix(dist)
    mst = minimum_spanning_tree(sparse)
    return float(mst.sum())
