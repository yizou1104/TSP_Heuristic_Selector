"""TSPLIB .tsp file parser — returns distance matrix and instance metadata."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import tsplib95


# Maps tsplib95 edge_weight_type strings to the type labels used during training.
_TYPE_MAP: dict[str, str] = {
    "EUC_2D": "EUC_2D",
    "EUC_3D": "EUC_2D",   # treat as Euclidean
    "GEO": "Geo",
    "ATT": "EUC_2D",       # pseudo-Euclidean, treat similarly
    "CEIL_2D": "EUC_2D",
    "EXPLICIT": "Explicit",
    "FULL_MATRIX": "Explicit",
    "UPPER_ROW": "Explicit",
    "LOWER_ROW": "Explicit",
    "UPPER_DIAG_ROW": "Explicit",
    "LOWER_DIAG_ROW": "Explicit",
    "UPPER_COL": "Explicit",
    "LOWER_COL": "Explicit",
    "UPPER_DIAG_COL": "Explicit",
    "LOWER_DIAG_COL": "Explicit",
}


def _euc2d_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return int(round(math.hypot(x1 - x2, y1 - y2)))


def _geo_lat_lon(coord: tuple[float, float]) -> tuple[float, float]:
    """Convert TSPLIB GEO format degree-minutes to radians."""
    deg = int(coord[0])
    minutes = coord[0] - deg
    lat = math.pi * (deg + 5.0 * minutes / 3.0) / 180.0
    deg = int(coord[1])
    minutes = coord[1] - deg
    lon = math.pi * (deg + 5.0 * minutes / 3.0) / 180.0
    return lat, lon


def _geo_distance(c1: tuple[float, float], c2: tuple[float, float]) -> int:
    """Geodesic distance per TSPLIB specification."""
    RRR = 6378.388
    lat1, lon1 = _geo_lat_lon(c1)
    lat2, lon2 = _geo_lat_lon(c2)
    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    return int(RRR * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0)


def load_distance_matrix(tsp_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a TSPLIB .tsp file and return (distance_matrix, metadata).

    distance_matrix: float64 ndarray of shape (n, n)
    metadata keys:
      - name: str
      - n: int
      - edge_type: str  one of 'EUC_2D', 'Geo', 'Explicit'
    """
    tsp_path = Path(tsp_path)
    problem = tsplib95.load(str(tsp_path))

    nodes = sorted(problem.get_nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    ewt = (problem.edge_weight_type or "EUC_2D").upper().strip()
    edge_type = _TYPE_MAP.get(ewt, "EUC_2D")

    dist = np.zeros((n, n), dtype=np.float64)

    if ewt == "GEO":
        coords = problem.node_coords
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j:
                    dist[i, j] = _geo_distance(coords[u], coords[v])

    elif ewt in ("EXPLICIT", "FULL_MATRIX", "UPPER_ROW", "LOWER_ROW",
                  "UPPER_DIAG_ROW", "LOWER_DIAG_ROW",
                  "UPPER_COL", "LOWER_COL",
                  "UPPER_DIAG_COL", "LOWER_DIAG_COL"):
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j:
                    dist[i, j] = float(problem.get_weight(u, v))

    else:
        # EUC_2D, ATT, CEIL_2D — use tsplib95 weight function which handles rounding
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j:
                    dist[i, j] = float(problem.get_weight(u, v))

    name = problem.name or tsp_path.stem

    return dist, {"name": name, "n": n, "edge_type": edge_type}
