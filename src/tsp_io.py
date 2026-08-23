"""Loading TSPLIB / plain-matrix / CSV instances, distance-matrix construction,
TSPLIB-EXPLICIT writing, and synthetic instance generators.

Extracted (with light cleanup) from the original notebook's I/O cells
(original lines ~40-420 for loading/distance-matrix code, ~2054-2148 for the
generators and TSPLIB writer). Functionality is unchanged from the notebook;
only the hardcoded Colab paths have been replaced with parameters that
default to the :mod:`src.config` constants.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tsplib95

from .config import (
    GENERATED_CHRISTOFIDES_HARD_DIR,
    GENERATED_CLUSTERED_DIR,
    GENERATED_RANDOM_DIR,
    RAW_EUCLIDEAN_2D_DIR,
    RAW_EXPLICIT_DIR,
    RAW_GEO_DIR,
)

EARTH_RADIUS = 6378.388  # TSPLIB GEO constant


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_tsplib_instance(path) -> dict:
    """Primary loader: use tsplib95 for TSPLIB-formatted .tsp files.

    Works for coordinate-based instances (e.g. att48) and explicit
    distance-matrix instances (e.g. gr17, fri26, dantzig42).
    """
    problem = tsplib95.load(str(path))
    nodes = list(problem.get_nodes())
    n = len(nodes)

    coords = None
    if getattr(problem, "node_coords", None):
        coords = dict(problem.node_coords.items())
    elif getattr(problem, "display_data", None):
        coords = dict(problem.display_data.items())

    dist = [[problem.get_weight(i, j) for j in nodes] for i in nodes]

    return {
        "name": problem.name,
        "nodes": nodes,
        "dimension": n,
        "edge_weight_type": problem.edge_weight_type,
        "coords": coords,
        "dist": dist,
        "problem": problem,
    }


def load_tsplib_matrix_fallback(path) -> dict:
    """Fallback for 'broken' TSPLIB matrix files where tsplib95 fails.

    Reads DIMENSION from the header, finds EDGE_WEIGHT_SECTION, collects
    numeric tokens until EOF, and interprets them as a FULL_MATRIX. No
    coordinates; this is matrix-only.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    dim = None
    in_edge = False
    vals = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        u = s.upper()

        if u.startswith("DIMENSION"):
            if ":" in s:
                dim = int(s.split(":", 1)[1])
            else:
                dim = int(s.split()[1])
        elif u.startswith("EDGE_WEIGHT_SECTION"):
            in_edge = True
            continue
        elif u.startswith("EOF"):
            break
        elif in_edge:
            for tok in s.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass

    if dim is None:
        raise ValueError(f"Could not find DIMENSION in {path}")

    n = dim
    needed = n * n
    if len(vals) < needed:
        raise ValueError(
            f"Not enough numbers in EDGE_WEIGHT_SECTION of {path}: "
            f"got {len(vals)}, need {needed}"
        )

    vals = vals[:needed]
    dist = [[0.0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(n):
            dist[i][j] = vals[idx]
            idx += 1

    nodes = list(range(1, n + 1))

    return {
        "name": os.path.basename(str(path)),
        "nodes": nodes,
        "dimension": n,
        "edge_weight_type": "EXPLICIT",
        "coords": None,
        "dist": dist,
        "problem": None,
    }


def load_plain_matrix(path) -> dict:
    """Load a plain whitespace-separated square distance matrix file."""
    mat = np.loadtxt(path)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{path} is not a square matrix")

    n = mat.shape[0]
    nodes = list(range(1, n + 1))

    return {
        "name": os.path.basename(str(path)),
        "nodes": nodes,
        "dimension": n,
        "edge_weight_type": "EXPLICIT",
        "coords": None,
        "dist": mat.tolist(),
        "problem": None,
    }


def load_csv_tsp(path) -> Optional[dict]:
    """Universal CSV loader for TSP coordinates.

    Supports id/x/y format, lat/long or city/node column names, headerless
    2-column CSVs, and CSVs with extra columns. Returns a dict
    {node: (x, y)}, or None if unusable.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("city") or cols.get("node")
    x_col = cols.get("x") or cols.get("longitude") or cols.get("lng")
    y_col = cols.get("y") or cols.get("latitude") or cols.get("lat")

    if x_col and y_col:
        if not id_col:
            df["__id"] = range(1, len(df) + 1)
            id_col = "__id"
        coords = {
            int(df[id_col].iloc[i]): (float(df[x_col].iloc[i]), float(df[y_col].iloc[i]))
            for i in range(len(df))
        }
        return coords

    try:
        df2 = pd.read_csv(path, header=None)
    except Exception:
        return None

    numeric_cols = [i for i in range(df2.shape[1]) if pd.api.types.is_numeric_dtype(df2[i])]
    if len(numeric_cols) >= 2:
        x_i, y_i = numeric_cols[:2]
        coords = {i + 1: (float(df2.loc[i, x_i]), float(df2.loc[i, y_i])) for i in range(len(df2))}
        return coords

    return None


def load_tsp_folder(directory, max_n: Optional[int] = None) -> dict:
    """Load every ``*.tsp`` file in ``directory`` into a dict of instances.

    Reconstructed helper (not present verbatim in the notebook, which instead
    had several ad hoc, one-off folder-loading loops with hardcoded Colab
    paths — see original lines ~224-317). ``max_n`` optionally skips
    instances larger than that dimension, mirroring the notebook's
    ``MAX_N`` / ``MAX_FILE_SIZE_MB`` guards.
    """
    directory = Path(directory)
    instances: dict[str, dict] = {}
    if not directory.is_dir():
        return instances

    for path in sorted(directory.glob("*.tsp")):
        try:
            inst = load_tsplib_instance(path)
        except Exception:
            try:
                inst = load_tsplib_matrix_fallback(path)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Skipped {path.name}: {exc}")
                continue

        if max_n is not None and inst["dimension"] > max_n:
            print(f"Skipped {path.name}: n={inst['dimension']} > {max_n}")
            continue

        inst["_source_path"] = path
        instances[path.stem] = inst

    return instances


# ---------------------------------------------------------------------------
# GEO distance (TSPLIB GEO edge-weight type)
# ---------------------------------------------------------------------------
def geo_to_radians(x: float) -> float:
    """Convert a TSPLIB GEO coordinate (DDD.MM) to radians."""
    deg = int(x)
    minutes = x - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0


def geo_distance(coord1, coord2) -> int:
    """TSPLIB GEO distance between two (lat, lon) points."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1 = geo_to_radians(lat1)
    lon1 = geo_to_radians(lon1)
    lat2 = geo_to_radians(lat2)
    lon2 = geo_to_radians(lon2)

    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)

    return int(EARTH_RADIUS * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1)


def geo_distance_matrix(inst: dict) -> np.ndarray:
    coords = inst["coords"]
    nodes = inst["nodes"]
    n = len(nodes)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = geo_distance(coords[nodes[i]], coords[nodes[j]])
    return dist


def get_distance_matrix(inst: dict) -> np.ndarray:
    """Return the NxN distance matrix for a loaded instance.

    Prefers an already-provided ``dist`` (EXPLICIT instances / most
    loaders), falls back to TSPLIB GEO distance, and finally to rounded
    Euclidean distance for coordinate-based instances.
    """
    if inst.get("dist") is not None:
        return np.array(inst["dist"])

    if inst["edge_weight_type"] == "GEO":
        if "edge_weight_matrix" not in inst:
            inst["edge_weight_matrix"] = geo_distance_matrix(inst)
        return inst["edge_weight_matrix"]

    coords = inst["coords"]
    nodes = inst["nodes"]
    n = len(nodes)
    dist = np.zeros((n, n))
    for i in range(n):
        x1, y1 = coords[nodes[i]]
        for j in range(n):
            x2, y2 = coords[nodes[j]]
            dist[i, j] = int(round(math.hypot(x1 - x2, y1 - y2)))
    return dist


# ---------------------------------------------------------------------------
# Writing / generating synthetic EXPLICIT instances
# ---------------------------------------------------------------------------
def write_tsplib_explicit(filepath, dist_matrix, name: str, comment: str = "") -> None:
    """Write an integer FULL_MATRIX TSPLIB EXPLICIT instance file."""
    n = len(dist_matrix)
    with open(filepath, "w") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"COMMENT: {comment}\n")
        f.write(f"DIMENSION: {n}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(n):
            row = " ".join(str(int(dist_matrix[i][j])) for j in range(n))
            f.write(row + "\n")
        f.write("EOF\n")


def gen_random_explicit(n: int, low: int = 1, high: int = 10000, seed: Optional[int] = None) -> np.ndarray:
    """Symmetric random integer distance matrix (uniform, non-metric)."""
    rng = np.random.default_rng(seed)
    w = rng.integers(low, high, size=(n, n))
    w = (w + w.T) // 2
    np.fill_diagonal(w, 0)
    return w


def gen_clustered_explicit(
    n: int,
    k_clusters: int = 4,
    intra_range=(1, 100),
    inter_range=(1500, 4000),
    seed: Optional[int] = None,
) -> np.ndarray:
    """Symmetric distance matrix with cheap intra-cluster / expensive inter-cluster edges."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n), dtype=int)

    sizes = [n // k_clusters] * k_clusters
    for i in range(n % k_clusters):
        sizes[i] += 1

    clusters = []
    idx = 0
    for s in sizes:
        clusters.append(list(range(idx, idx + s)))
        idx += s

    for i in range(n):
        for j in range(i + 1, n):
            same = any(i in c and j in c for c in clusters)
            if same:
                w_ij = rng.integers(*intra_range)
            else:
                w_ij = rng.integers(*inter_range)
            w[i, j] = w[j, i] = w_ij

    return w


def gen_christofides_hard(n: int, seed: Optional[int] = None) -> np.ndarray:
    """Adversarial instance designed to make Christofides/MST-based reasoning
    misleading: two cheap intra-cluster halves, expensive inter-cluster
    edges, and a handful of deliberately cheap cross edges that poison the
    minimum spanning tree.
    """
    rng = np.random.default_rng(seed)

    w = rng.integers(800, 1200, size=(n, n))
    w = (w + w.T) // 2
    np.fill_diagonal(w, 0)

    split = n // 2

    for i in range(split):
        for j in range(split):
            if i != j:
                w[i, j] = rng.integers(1, 40)

    for i in range(split, n):
        for j in range(split, n):
            if i != j:
                w[i, j] = rng.integers(1, 40)

    for i in range(split):
        for j in range(split, n):
            w[i, j] = w[j, i] = rng.integers(2500, 5000)

    for _ in range(n // 2):
        i = rng.integers(0, split)
        j = rng.integers(split, n)
        w[i, j] = w[j, i] = rng.integers(5, 20)

    return w


# Generator family registry, mirroring the notebook's ``families`` dict
# (original line ~2153), now keyed by the same names as the
# ``data/generated/<family>`` directories.
GENERATOR_FAMILIES = {
    "random": gen_random_explicit,
    "clustered": gen_clustered_explicit,
    "christofides_hard": gen_christofides_hard,
}

GENERATOR_OUTPUT_DIRS = {
    "random": GENERATED_RANDOM_DIR,
    "clustered": GENERATED_CLUSTERED_DIR,
    "christofides_hard": GENERATED_CHRISTOFIDES_HARD_DIR,
}


# ---------------------------------------------------------------------------
# Convenience wrappers defaulting to the standard data/raw and data/generated
# subdirectories declared in src.config.
# ---------------------------------------------------------------------------
def load_raw_euclidean_2d(directory=RAW_EUCLIDEAN_2D_DIR, max_n: Optional[int] = None) -> dict:
    return load_tsp_folder(directory, max_n=max_n)


def load_raw_explicit(directory=RAW_EXPLICIT_DIR, max_n: Optional[int] = None) -> dict:
    return load_tsp_folder(directory, max_n=max_n)


def load_raw_geo(directory=RAW_GEO_DIR, max_n: Optional[int] = None) -> dict:
    return load_tsp_folder(directory, max_n=max_n)


def load_generated(family: str, max_n: Optional[int] = None) -> dict:
    """Load all generated instances for one family ('random', 'clustered',
    or 'christofides_hard'), from its ``data/generated/<family>`` directory.
    """
    directory = GENERATOR_OUTPUT_DIRS[family]
    return load_tsp_folder(directory, max_n=max_n)
