"""Build ``data/dataset.csv`` from raw + generated TSP instances.

RECONSTRUCTED end-to-end orchestrator. The original notebook never assembles
this long-format table in one place: it starts every "Engineering Features"
and "Constructing Explicit Instances" cell from an *already existing*
``Dataset.csv`` on Google Drive and repeatedly does "read CSV, add one
column, write CSV" (see original lines ~1360-2045, one such round-trip per
feature). This module replaces all of that with a single in-memory build and
one final write, while faithfully reusing the notebook's actual algorithms
for each step:

  - heuristics + runtime timing            -> src/heuristics.py
  - feature extraction                     -> src/features.py
  - LKH subprocess + .par file             -> original lines ~2223-2282
  - Concorde subprocess                    -> original lines ~2284-2352
  - solutions.txt parsing                  -> original lines ~2373-2384
  - optimality_gap formula + sentinel      -> original lines ~2397-2438
  - best-heuristic label (argmin cost)     -> equivalent to the
        `sort_values(["file","cost"]).groupby("file").first()` pattern used
        throughout the notebook's final "K-Fold Stratified Models" section
        (e.g. original lines ~6809-6826), simplified to a direct argmin
        per the task spec (the notebook's EPS=0.05 "is_best" tolerance mask
        is NOT carried forward — see module docstring note below).

Note on labeling: the notebook computes ``is_best = feasible & (cost <=
1.05 * best_feasible_cost)`` and then does
``sort_values(["file","cost"]).groupby("file").first()``. Because sorting by
cost ascending and taking the first row per file always yields the row with
the single smallest cost (which is always included in its own 5% tolerance
band), the net effect is identical to a plain single-label argmin. We
implement the argmin directly here instead of reconstructing the unused
tolerance machinery.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    CONCORDE_PATH,
    DATASET_PATH,
    LKH_PATH,
    RAW_EUCLIDEAN_2D_DIR,
    RAW_EXPLICIT_DIR,
    RAW_GEO_DIR,
    SOLUTIONS_TXT_PATH,
)
from .features import extract_all_features
from .heuristics import HEURISTICS
from .tsp_io import GENERATOR_OUTPUT_DIRS, get_distance_matrix, load_tsp_folder

# Runtime cap (ms) used to decide feasibility for the best-heuristic label
# and for the optimality-gap sentinel check (original: RUNTIME_CAP = 600_000,
# used throughout the notebook's model-training cells, e.g. original line
# ~6801).
RUNTIME_CAP_MS = 600_000

# Sentinel runtime (ms) recorded when a heuristic hard-fails or is skipped
# outright (too large to run at all), distinct from RUNTIME_CAP_MS which
# just marks "too slow to count as feasible". Mirrors the notebook's
# ``runtime_ms != 1000000`` sentinel check (original line ~2426).
TIMEOUT_SENTINEL_MS = 1_000_000

# Notebook size guards for the heuristics that don't scale well (original
# lines ~945 MAX_INSERTION_N=2000; christofides had MAX_CHRISTOFIDES_N=3000
# at original line ~698). NN and Greedy have no such guard in the notebook.
MAX_INSERTION_N = 2000
MAX_CHRISTOFIDES_N = 3000
MAX_HEURISTIC_N = {"Insertion": MAX_INSERTION_N, "CH": MAX_CHRISTOFIDES_N}


def classify_edge_weight_type(edge_weight_type: str) -> str:
    """Map a TSPLIB EDGE_WEIGHT_TYPE onto the dataset's 3-way `type` column
    (EUC_2D / Explicit / Geo).
    """
    ewt = (edge_weight_type or "").upper()
    if ewt == "GEO":
        return "Geo"
    if ewt in ("EXPLICIT",):
        return "Explicit"
    return "EUC_2D"


def load_all_instances(max_n: Optional[int] = None) -> dict[str, dict]:
    """Load every raw instance (euclidean_2d / explicit / geo) plus every
    generated instance (random / clustered / christofides_hard).
    """
    instances: dict[str, dict] = {}
    for directory in (RAW_EUCLIDEAN_2D_DIR, RAW_EXPLICIT_DIR, RAW_GEO_DIR):
        instances.update(load_tsp_folder(directory, max_n=max_n))
    for family, directory in GENERATOR_OUTPUT_DIRS.items():
        instances.update(load_tsp_folder(directory, max_n=max_n))
    return instances


def run_heuristics_on_instance(name: str, inst: dict) -> list[dict]:
    """Run all 4 heuristics on one instance, returning one row dict per
    heuristic with file/type/n/heuristic/cost/runtime_ms.
    """
    dist = get_distance_matrix(inst)
    n = dist.shape[0]
    inst_type = classify_edge_weight_type(inst.get("edge_weight_type"))

    rows = []
    for heuristic_name, heuristic_fn in HEURISTICS.items():
        size_limit = MAX_HEURISTIC_N.get(heuristic_name)
        if size_limit is not None and n > size_limit:
            rows.append(
                {
                    "file": name,
                    "type": inst_type,
                    "n": n,
                    "heuristic": heuristic_name,
                    "cost": np.nan,
                    "runtime_ms": TIMEOUT_SENTINEL_MS,
                }
            )
            continue

        try:
            start = time.perf_counter()
            _, cost = heuristic_fn(dist)
            runtime_ms = (time.perf_counter() - start) * 1000.0
        except Exception:
            cost = np.nan
            runtime_ms = TIMEOUT_SENTINEL_MS

        rows.append(
            {
                "file": name,
                "type": inst_type,
                "n": n,
                "heuristic": heuristic_name,
                "cost": cost,
                "runtime_ms": runtime_ms,
            }
        )
    return rows


def build_feature_rows(instances: dict[str, dict]) -> pd.DataFrame:
    """One row per instance with its 10 Stage-1 features + `file`/`type`."""
    rows = []
    for name, inst in instances.items():
        dist = get_distance_matrix(inst)
        feats = extract_all_features(dist)
        feats["file"] = name
        feats["type"] = classify_edge_weight_type(inst.get("edge_weight_type"))
        rows.append(feats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optimal-cost lookup: solutions.txt, then LKH, then Concorde.
# ---------------------------------------------------------------------------
def load_optimal_solutions(filepath=SOLUTIONS_TXT_PATH) -> dict[str, float]:
    """Parse a `name: value` solutions file (one known/computed optimal tour
    cost per line) into a dict. Original lines ~2373-2384.
    """
    filepath = Path(filepath)
    optimal: dict[str, float] = {}
    if not filepath.exists():
        return optimal
    with open(filepath, "r") as f:
        for line in f:
            if ":" in line:
                name, value = line.split(":", 1)
                name = name.strip()
                value = value.strip().split()[0]  # strip trailing annotations
                optimal[name] = float(value)
    return optimal


def run_lkh(tsp_path, name: str, runs: int = 20, lkh_path=LKH_PATH, work_dir: Optional[Path] = None) -> Optional[float]:
    """Run LKH-3 on one instance via a generated .par file and return the
    best tour cost found, or None if it could not be parsed. Original lines
    ~2251-2278.
    """
    work_dir = Path(work_dir) if work_dir is not None else Path(tsp_path).parent
    par_path = work_dir / f"{name}.par"
    tour_path = work_dir / f"{name}.tour"
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
        f.write(f"RUNS = {runs}\n")

    result = subprocess.run(
        [str(lkh_path), str(par_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    match = re.search(r"Cost = (\d+)", result.stdout)
    return float(match.group(1)) if match else None


def run_concorde(tsp_path, concorde_path=CONCORDE_PATH) -> Optional[float]:
    """Run Concorde on one instance and return the optimal tour cost, or
    None if it could not be parsed. Original lines ~2284-2346.
    """
    result = subprocess.run(
        [str(concorde_path), str(tsp_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    match = re.search(r"(Optimal.*?)(\d+)", result.stdout, re.IGNORECASE)
    return float(match.group(2)) if match else None


def compute_optimal_costs(
    instance_names: list[str],
    tsp_paths: dict[str, Path],
    solutions: Optional[dict[str, float]] = None,
    use_lkh: bool = True,
    use_concorde: bool = False,
) -> dict[str, float]:
    """Resolve an optimal (or best-known) cost per instance name.

    Lookup order: pre-solved ``solutions.txt`` entries first (covers real
    TSPLIB instances with published optima), then LKH (near-optimal, fast,
    used for generated instances in the notebook), then Concorde (exact,
    slow) if explicitly requested.
    """
    solutions = dict(solutions or {})
    optimal_costs: dict[str, float] = {}

    for name in instance_names:
        if name in solutions:
            optimal_costs[name] = solutions[name]
            continue

        tsp_path = tsp_paths.get(name)
        if tsp_path is None:
            continue

        cost = None
        if use_lkh:
            cost = run_lkh(tsp_path, name)
        if cost is None and use_concorde:
            cost = run_concorde(tsp_path)
        if cost is not None:
            optimal_costs[name] = cost

    return optimal_costs


# ---------------------------------------------------------------------------
# Labeling + optimality gap
# ---------------------------------------------------------------------------
def add_best_heuristic_label(df: pd.DataFrame, runtime_cap_ms: float = RUNTIME_CAP_MS) -> pd.DataFrame:
    """Add `feasible`, `best_feasible_cost`, and `best_heuristic` columns.

    `best_heuristic` is the single-label argmin-cost heuristic among the
    feasible (runtime_ms <= runtime_cap_ms) rows for each instance — see the
    module docstring for why this is equivalent to, and simpler than, the
    notebook's EPS=0.05 tolerance-mask + sort/first construction.
    """
    df = df.copy()
    df["feasible"] = df["runtime_ms"] <= runtime_cap_ms
    df["best_feasible_cost"] = df[df["feasible"]].groupby("file")["cost"].transform("min")

    feasible_rows = df[df["feasible"] & df["cost"].notna()]
    best_idx = feasible_rows.groupby("file")["cost"].idxmin()
    best_heuristic = df.loc[best_idx, ["file", "heuristic"]].set_index("file")["heuristic"]
    df["best_heuristic"] = df["file"].map(best_heuristic)
    return df


def add_optimality_gap(df: pd.DataFrame, timeout_sentinel_ms: float = TIMEOUT_SENTINEL_MS) -> pd.DataFrame:
    """Add `optimality_gap` = (cost - optimal_cost) / optimal_cost * 100,
    for rows with a known cost, a known optimal_cost, and a non-sentinel
    runtime. Original lines ~2397-2438.
    """
    df = df.copy()
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["optimal_cost"] = pd.to_numeric(df["optimal_cost"], errors="coerce")

    df["optimality_gap"] = np.nan
    valid = (
        df["cost"].notna()
        & df["optimal_cost"].notna()
        & (df["optimal_cost"] != 0)
        & (df["runtime_ms"] != timeout_sentinel_ms)
    )
    df.loc[valid, "optimality_gap"] = (
        (df.loc[valid, "cost"] - df.loc[valid, "optimal_cost"]) / df.loc[valid, "optimal_cost"]
    ) * 100
    return df


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def build_dataset(
    max_n: Optional[int] = None,
    use_lkh: bool = True,
    use_concorde: bool = False,
    output_path=DATASET_PATH,
) -> pd.DataFrame:
    """Load all instances, run all 4 heuristics on each, extract features,
    resolve optimal costs, compute labels + gap, and write dataset.csv.
    """
    instances = load_all_instances(max_n=max_n)
    if not instances:
        raise RuntimeError(
            "No .tsp instances found under data/raw/* or data/generated/*. "
            "Populate those directories before running build_dataset()."
        )

    heuristic_rows = []
    for name, inst in instances.items():
        heuristic_rows.extend(run_heuristics_on_instance(name, inst))
    df = pd.DataFrame(heuristic_rows)

    feature_df = build_feature_rows(instances)
    df = df.merge(feature_df, on=["file", "type"], how="left")

    tsp_paths = {name: inst.get("_source_path") for name, inst in instances.items()}
    solutions = load_optimal_solutions()
    optimal_costs = compute_optimal_costs(
        list(instances.keys()), tsp_paths, solutions=solutions, use_lkh=use_lkh, use_concorde=use_concorde
    )
    df["optimal_cost"] = df["file"].map(optimal_costs)

    df = add_best_heuristic_label(df)
    df = add_optimality_gap(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    build_dataset()
