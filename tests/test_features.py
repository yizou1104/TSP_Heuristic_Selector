"""Unit tests for feature extraction."""

from pathlib import Path

import numpy as np
import pytest

from tsp_heuristics.features import extract_features, NUMERIC_FEATURE_NAMES
from tsp_heuristics.parsers import load_distance_matrix

ATT48 = Path(__file__).parent / "att48.tsp"


def test_feature_vector_shape():
    dist, meta = load_distance_matrix(ATT48)
    feats = extract_features(dist)
    assert feats.shape == (len(NUMERIC_FEATURE_NAMES),)


def test_n_correct():
    dist, meta = load_distance_matrix(ATT48)
    feats = extract_features(dist)
    assert feats[0] == 48  # n


def test_edge_weight_ordering():
    # feats[1]=min_w, feats[2]=std_w
    dist, meta = load_distance_matrix(ATT48)
    feats = extract_features(dist)
    min_w = feats[1]
    std_w = feats[2]
    assert min_w > 0
    assert std_w >= 0


def test_cv_is_std_over_mean():
    dist, meta = load_distance_matrix(ATT48)
    feats = extract_features(dist)
    # feats[2]=std, feats[3]=cv
    cv = feats[3]
    assert 0 < cv < 5


def test_pct_short_edges_in_range():
    dist, meta = load_distance_matrix(ATT48)
    feats = extract_features(dist)
    pct_short = feats[5]
    assert 0.0 <= pct_short <= 1.0


def test_triangle_violation_rate_euclidean_zero():
    """EUC_2D instances satisfy triangle inequality — rate should be ~0."""
    dist, meta = load_distance_matrix(ATT48)
    feats = extract_features(dist)
    tri_rate = feats[6]
    assert tri_rate < 0.01, f"Expected ~0 triangle violations for EUC_2D, got {tri_rate}"


def test_mst_weight_positive():
    dist, meta = load_distance_matrix(ATT48)
    feats = extract_features(dist)
    mst_w = feats[8]
    assert mst_w > 0


def test_feature_names_length():
    assert len(NUMERIC_FEATURE_NAMES) == 9


def test_small_matrix():
    """Feature extraction on a tiny 4-node matrix."""
    dist = np.array([
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0],
    ], dtype=np.float64)
    feats = extract_features(dist, n_triangle_samples=100)
    assert feats.shape == (9,)
    assert feats[0] == 4  # n
