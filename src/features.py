"""Instance structural features used by the Stage 1 / Stage 2 models.

Each ``compute_*`` function takes a distance matrix (or, where noted, a
precomputed intermediate) and returns a single feature value. Extracted from
the notebook's "Engineering Features" section (original lines ~1327-2045).

Final Stage-1 feature set (10 features, per the paper):
    type (categorical, handled outside this module), n, min_edge_weight,
    std_edge_weight, cv_edge_weight, edge_weight_skewness, pct_short_edges,
    triangle_violation_rate, avg_violation_magnitude, mst_weight

``max_edge_weight`` and ``nn_cost_over_mst`` appear in the notebook but were
dropped from the final feature set (see original lines ~1395-1450 and
~2010-2045); they are intentionally NOT exposed here as model features.
``compute_mean_edge_weight`` is kept only as the internal intermediate
needed by ``cv_edge_weight``, ``edge_weight_skewness``, and the mst_weight
normalisation below — it is not itself one of the 10 model inputs.

IMPORTANT (bug fix carried over from the notebook's final "K-Fold Stratified
Models" / "XGBoost Tuned" section, original lines ~9658-9659, ~10021-10022):
``mst_weight`` must be normalised by ``n * mean_edge_weight`` exactly ONCE,
at dataset-construction time (see ``extract_all_features`` below and
``build_dataset.py``). Do not divide by ``n * mean_edge_weight`` again
anywhere downstream (e.g. in models.py) — the notebook had an earlier,
double-normalization-prone draft of this step which this reconstruction
does not carry forward.
"""

from __future__ import annotations

import math
import random

import numpy as np

# Sample size used for the two triangle-inequality features, matching the
# notebook's "Triangle_Violation_Rate (Sample Size 200000)" section
# (original lines ~1785, ~1850).
TRIANGLE_SAMPLE_SIZE = 200_000


def _upper_triangle_weights(dist_matrix) -> np.ndarray:
    """Flat array of the n*(n-1)/2 distinct edge weights (i < j)."""
    dist_matrix = np.asarray(dist_matrix)
    n = dist_matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    return dist_matrix[iu]


def compute_min_edge_weight(dist_matrix) -> float:
    weights = _upper_triangle_weights(dist_matrix)
    positive = weights[weights > 0]
    if positive.size == 0:
        return 0.0
    return float(positive.min())


def compute_max_edge_weight(dist_matrix) -> float:
    """Maximum edge weight. NOT part of the final 10-feature Stage-1 set —
    kept here only for completeness/analysis, matching the notebook function
    (original lines ~1395-1406) that this feature set later dropped.
    """
    weights = _upper_triangle_weights(dist_matrix)
    return float(weights.max())


def compute_mean_edge_weight(dist_matrix) -> float:
    """Internal intermediate only — not a Stage-1 model feature itself, but
    required to compute cv_edge_weight, edge_weight_skewness, and the
    mst_weight normalisation.
    """
    weights = _upper_triangle_weights(dist_matrix)
    return float(weights.mean())


def compute_std_edge_weight(dist_matrix, mean_edge_weight: float | None = None) -> float:
    """Population standard deviation of edge weights."""
    weights = _upper_triangle_weights(dist_matrix)
    mean = compute_mean_edge_weight(dist_matrix) if mean_edge_weight is None else mean_edge_weight
    var = float(np.mean((weights - mean) ** 2))
    return math.sqrt(var)


def compute_cv_edge_weight(dist_matrix, mean_edge_weight: float | None = None, std_edge_weight: float | None = None) -> float:
    """Coefficient of variation: std / mean."""
    mean = compute_mean_edge_weight(dist_matrix) if mean_edge_weight is None else mean_edge_weight
    if mean == 0:
        return 0.0
    std = compute_std_edge_weight(dist_matrix, mean_edge_weight=mean) if std_edge_weight is None else std_edge_weight
    return std / mean


def compute_edge_weight_skewness(dist_matrix, mean_edge_weight: float | None = None, std_edge_weight: float | None = None) -> float:
    """Population (Fisher) skewness of the edge-weight distribution."""
    weights = _upper_triangle_weights(dist_matrix)
    mean = compute_mean_edge_weight(dist_matrix) if mean_edge_weight is None else mean_edge_weight
    std = compute_std_edge_weight(dist_matrix, mean_edge_weight=mean) if std_edge_weight is None else std_edge_weight
    if std == 0:
        return 0.0
    z = (weights - mean) / std
    return float(np.mean(z ** 3))


def compute_pct_short_edges(dist_matrix, percentile: float = 10) -> float:
    """Fraction of edges at or below the given percentile of edge weights
    (default: 10th percentile, i.e. the "short" edges)."""
    weights = _upper_triangle_weights(dist_matrix)
    threshold = np.percentile(weights, percentile)
    return float(np.mean(weights <= threshold))


def compute_triangle_violation_rate(dist_matrix, num_samples: int = TRIANGLE_SAMPLE_SIZE, seed: int = 42) -> float:
    """Fraction of sampled (i, j, k) triples violating the triangle
    inequality dist(i,k) <= dist(i,j) + dist(j,k).
    """
    dist_matrix = np.asarray(dist_matrix)
    n = dist_matrix.shape[0]
    if n < 3:
        return 0.0

    rng = random.Random(seed)
    violations = 0
    for _ in range(num_samples):
        i, j, k = rng.sample(range(n), 3)
        dij = dist_matrix[i, j]
        djk = dist_matrix[j, k]
        dik = dist_matrix[i, k]
        if dik > dij + djk:
            violations += 1

    return violations / num_samples


def compute_avg_violation_magnitude(dist_matrix, num_samples: int = TRIANGLE_SAMPLE_SIZE, seed: int = 42) -> float:
    """Mean excess (dik - (dij + djk)) over sampled triples that violate the
    triangle inequality; 0.0 if none are sampled or n < 3.
    """
    dist_matrix = np.asarray(dist_matrix)
    n = dist_matrix.shape[0]
    if n < 3:
        return 0.0

    rng = random.Random(seed)
    total_violation = 0.0
    violations = 0
    for _ in range(num_samples):
        i, j, k = rng.sample(range(n), 3)
        dij = dist_matrix[i, j]
        djk = dist_matrix[j, k]
        dik = dist_matrix[i, k]
        violation = dik - (dij + djk)
        if violation > 0:
            total_violation += violation
            violations += 1

    if violations == 0:
        return 0.0
    return total_violation / violations


def compute_mst_weight_raw(dist_matrix) -> float:
    """Total weight of the minimum spanning tree (Prim's algorithm, O(n^2)).

    This is the RAW (un-normalised) MST weight. Use
    ``extract_all_features`` (or divide by ``n * mean_edge_weight`` exactly
    once yourself) to get the final normalised ``mst_weight`` feature.
    """
    dist_matrix = np.asarray(dist_matrix)
    n = dist_matrix.shape[0]

    in_mst = [False] * n
    min_edge = [math.inf] * n
    min_edge[0] = 0.0
    total_weight = 0.0

    for _ in range(n):
        u = -1
        best = math.inf
        for i in range(n):
            if not in_mst[i] and min_edge[i] < best:
                best = min_edge[i]
                u = i

        in_mst[u] = True
        total_weight += best

        for v in range(n):
            if not in_mst[v] and dist_matrix[u, v] < min_edge[v]:
                min_edge[v] = dist_matrix[u, v]

    return total_weight


def extract_all_features(dist_matrix) -> dict:
    """Compute the 10 Stage-1 structural features for one instance.

    ``type`` is intentionally excluded (it is metadata about the instance's
    TSPLIB edge-weight type, not derived from the distance matrix, and is
    supplied separately by the caller — see build_dataset.py).

    ``mst_weight`` is normalised by ``n * mean_edge_weight`` exactly once,
    here — this is the single correct place for that normalisation. Do not
    repeat it anywhere else in the pipeline.
    """
    dist_matrix = np.asarray(dist_matrix)
    n = dist_matrix.shape[0]

    mean_edge_weight = compute_mean_edge_weight(dist_matrix)
    std_edge_weight = compute_std_edge_weight(dist_matrix, mean_edge_weight=mean_edge_weight)
    cv_edge_weight = compute_cv_edge_weight(dist_matrix, mean_edge_weight=mean_edge_weight, std_edge_weight=std_edge_weight)
    edge_weight_skewness = compute_edge_weight_skewness(
        dist_matrix, mean_edge_weight=mean_edge_weight, std_edge_weight=std_edge_weight
    )
    mst_weight_raw = compute_mst_weight_raw(dist_matrix)
    mst_weight = mst_weight_raw / (n * mean_edge_weight)

    return {
        "n": n,
        "min_edge_weight": compute_min_edge_weight(dist_matrix),
        "std_edge_weight": std_edge_weight,
        "cv_edge_weight": cv_edge_weight,
        "edge_weight_skewness": edge_weight_skewness,
        "pct_short_edges": compute_pct_short_edges(dist_matrix),
        "triangle_violation_rate": compute_triangle_violation_rate(dist_matrix),
        "avg_violation_magnitude": compute_avg_violation_magnitude(dist_matrix),
        "mst_weight": mst_weight,
    }


# The final Stage-1 / Stage-2 feature column lists, shared by build_dataset.py
# and models.py so both stay in sync with the paper's spec.
STAGE1_INSTANCE_FEATURES = [
    "n",
    "min_edge_weight",
    "std_edge_weight",
    "cv_edge_weight",
    "edge_weight_skewness",
    "pct_short_edges",
    "triangle_violation_rate",
    "avg_violation_magnitude",
    "mst_weight",
    "type",
]
STAGE1_NUMERIC_FEATURES = [f for f in STAGE1_INSTANCE_FEATURES if f != "type"]
STAGE1_CATEGORICAL_FEATURES = ["type"]

# Stage 2 adds `heuristic` as an 11th (categorical) feature.
STAGE2_FEATURE_COLUMNS = STAGE1_INSTANCE_FEATURES + ["heuristic"]
STAGE2_CATEGORICAL_FEATURES = ["type", "heuristic"]
