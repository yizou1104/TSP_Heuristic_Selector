"""Timing benchmark: feature-extraction cost vs. heuristic-execution cost.

For every real ``.tsp`` instance available under ``data/raw/`` and
``data/generated/``, this measures (single wall-clock run each, via
``time.perf_counter()`` — NOT averaged over repeats):

  1. Feature extraction, split into two parts:
       - cheap_features:      everything in extract_all_features() except
                               triangle_violation_rate / avg_violation_magnitude
       - monte_carlo_features: those two O(TRIANGLE_SAMPLE_SIZE) functions
  2. Each of the 4 construction heuristics individually (NN, Greedy,
     Insertion, Christofides), for instances at or below HEURISTIC_N_CUTOFF
     (see the module docstring section below for why).
  3. Stage 1 + Stage 2 inference time: a SINGLE predict() (Stage 1 classifier)
     + predict_gap() (Stage 2 regressor) call on an ALREADY-FITTED RF+Quantile
     pipeline pair. NOTE: src/models.py has no persisted/serialized model
     artifacts (fit_random_forest / fit_gap_regressor_rf_quantile always fit
     from data in-memory, with no joblib.dump/load path) — retraining per
     benchmarked instance would be both unrepresentative of real inference
     cost and impractical time-wise, so we fit ONE representative RF+Quantile
     pipeline pair ONCE (outside the timed loop, on the existing
     data/Dataset.csv, SEED=1 matching run_integrated_pipeline's real 128/32
     instance split) and then time a single predict/predict_gap call per
     instance against that already-fitted pipeline. This is the documented
     fallback ("time a SINGLE representative predict call on an
     already-fitted small pipeline instead of full training").

CAVEAT: all timings are SINGLE-RUN wall-clock measurements on this machine,
not averaged over repeated runs. Heuristic runtimes in particular can vary
run to run (CPU scheduling noise, thermal throttling, other processes) more
than feature-extraction times do; treat these as one-shot estimates, not
tightly-controlled benchmarks.

HEURISTIC N CUTOFF
------------------
insertion_tsp_from_matrix and christofides_tsp are both O(n^3) with a large
constant factor in pure Python (insertion: naive triple-nested-loop cheapest
insertion; christofides: O(n^3) minimum-weight perfect matching via
networkx). An empirical scaling probe on this machine (random symmetric
matrices, single run each) gave:

    n     nn        greedy     insertion   christofides
    100   0.0009s   0.002s     0.048s      0.044s
    200   0.003s    0.011s     0.373s      0.297s
    400   0.012s    0.036s     2.986s      1.976s
    600   0.027s    0.083s     9.801s      6.192s
    800   0.047s    0.157s     24.964s     12.243s

Insertion alone already costs ~25s per instance at n=800 and grows cubically
from there (n=1400 would be ~130s+ for insertion alone). Running all 4
heuristics on every one of the 155 available instances — 25 of which have
n > 800, up to n=11849 — would make this benchmark impractical to complete
in one session. HEURISTIC_N_CUTOFF = 800 was chosen so that:
  - all 4 heuristics stay under ~25s each (single worst-case instance ~40s),
  - 130 of the 155 available instances (84%) are covered,
  - the remaining 25 large instances still get feature-extraction and
    Stage 1+2 inference timed and reported in the CSV (with heuristic/
    speedup columns left blank), so no data is silently dropped — they are
    simply excluded from the speedup calculation, which requires all 4
    heuristics to have run.

Usage:
    python experiments/timing_benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_dataset import classify_edge_weight_type  # noqa: E402
from src.config import DATASET_PATH  # noqa: E402
from src.features import (  # noqa: E402
    STAGE1_INSTANCE_FEATURES,
    compute_avg_violation_magnitude,
    compute_cv_edge_weight,
    compute_edge_weight_skewness,
    compute_mean_edge_weight,
    compute_min_edge_weight,
    compute_mst_weight_raw,
    compute_pct_short_edges,
    compute_std_edge_weight,
    compute_triangle_violation_rate,
)
from src.heuristics import (  # noqa: E402
    christofides_tsp,
    greedy_tsp_from_matrix,
    insertion_tsp_from_matrix,
    nearest_neighbour_tsp,
)
from src.models import (  # noqa: E402
    fit_gap_regressor_rf_quantile,
    fit_random_forest,
    prepare_stage2_split,
)
from src.tsp_io import (  # noqa: E402
    get_distance_matrix,
    load_generated,
    load_raw_euclidean_2d,
    load_raw_explicit,
    load_raw_geo,
)

HEURISTIC_N_CUTOFF = 800  # see module docstring for justification

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV_PATH = RESULTS_DIR / "timing_benchmark.csv"

SIZE_BUCKETS = [
    (0, 130, "small (n<=130)"),
    (131, 300, "medium (131-300)"),
    (301, 500, "large (301-500)"),
    (501, 800, "xlarge (501-800)"),
]

HEURISTICS = {
    "nn": nearest_neighbour_tsp,
    "greedy": greedy_tsp_from_matrix,
    "insertion": insertion_tsp_from_matrix,
    "christofides": christofides_tsp,
}


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------
def load_all_instances() -> dict:
    """Load every .tsp instance from data/raw/ and data/generated/, reusing
    the existing loaders in src/tsp_io.py. Returns {label: inst_dict}.
    """
    instances = {}

    for family_label, loader in [
        ("raw/euclidean_2d", load_raw_euclidean_2d),
        ("raw/explicit", load_raw_explicit),
        ("raw/geo", load_raw_geo),
    ]:
        for name, inst in loader().items():
            inst["_family"] = family_label
            instances[f"{family_label}/{name}"] = inst

    for family in ["random", "clustered", "christofides_hard"]:
        for name, inst in load_generated(family).items():
            inst["_family"] = f"generated/{family}"
            instances[f"generated/{family}/{name}"] = inst

    return instances


# ---------------------------------------------------------------------------
# Feature timing (mirrors extract_all_features' internal computation order
# exactly, so cheap_features + monte_carlo_features == what
# extract_all_features would compute, without a redundant 3rd full pass)
# ---------------------------------------------------------------------------
def timed_cheap_features(dist_matrix, n: int):
    t0 = time.perf_counter()
    mean_edge_weight = compute_mean_edge_weight(dist_matrix)
    std_edge_weight = compute_std_edge_weight(dist_matrix, mean_edge_weight=mean_edge_weight)
    cv_edge_weight = compute_cv_edge_weight(
        dist_matrix, mean_edge_weight=mean_edge_weight, std_edge_weight=std_edge_weight
    )
    edge_weight_skewness = compute_edge_weight_skewness(
        dist_matrix, mean_edge_weight=mean_edge_weight, std_edge_weight=std_edge_weight
    )
    min_edge_weight = compute_min_edge_weight(dist_matrix) / mean_edge_weight
    pct_short_edges = compute_pct_short_edges(dist_matrix)
    mst_weight_raw = compute_mst_weight_raw(dist_matrix)
    mst_weight = mst_weight_raw / (n * mean_edge_weight)
    elapsed = time.perf_counter() - t0

    features = {
        "n": n,
        "min_edge_weight": min_edge_weight,
        "std_edge_weight": std_edge_weight,
        "cv_edge_weight": cv_edge_weight,
        "edge_weight_skewness": edge_weight_skewness,
        "pct_short_edges": pct_short_edges,
        "mst_weight": mst_weight,
    }
    return features, elapsed, mean_edge_weight


def timed_monte_carlo_features(dist_matrix, mean_edge_weight: float):
    t0 = time.perf_counter()
    triangle_violation_rate = compute_triangle_violation_rate(dist_matrix)
    avg_violation_magnitude = compute_avg_violation_magnitude(dist_matrix) / mean_edge_weight
    elapsed = time.perf_counter() - t0

    features = {
        "triangle_violation_rate": triangle_violation_rate,
        "avg_violation_magnitude": avg_violation_magnitude,
    }
    return features, elapsed


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading instances from data/raw/ and data/generated/ ...")
    instances = load_all_instances()
    print(f"Loaded {len(instances)} instances.\n")

    if not Path(DATASET_PATH).exists():
        sys.exit(f"Dataset not found at {DATASET_PATH}. Build it via python -m src.build_dataset first.")

    print("Fitting ONE representative Stage 1 (RF) + Stage 2 (RF+Quantile) pipeline pair "
          f"once, from {DATASET_PATH} (SEED=1) ...")
    df = pd.read_csv(DATASET_PATH)
    split = prepare_stage2_split(df)
    X_train_class = split["best_rows_train"][STAGE1_INSTANCE_FEATURES]
    y_train_class = split["best_rows_train"]["heuristic"]
    pipeline_class = fit_random_forest(X_train_class, y_train_class)
    gap_fit = fit_gap_regressor_rf_quantile(split["df_gap"], split["train_files"])
    predict_gap = gap_fit["predict"]
    print("Done fitting. Starting per-instance timing...\n")

    rows = []
    skip_log = []  # (file, reason) for auditability

    for idx, (label, inst) in enumerate(sorted(instances.items(), key=lambda kv: kv[1]["dimension"]), start=1):
        n = inst["dimension"]
        file = Path(label).stem
        inst_type = classify_edge_weight_type(inst.get("edge_weight_type"))

        dist_matrix = get_distance_matrix(inst)

        cheap_feats, t_cheap, mean_edge_weight = timed_cheap_features(dist_matrix, n)
        mc_feats, t_mc = timed_monte_carlo_features(dist_matrix, mean_edge_weight)
        t_features_total = t_cheap + t_mc

        instance_features = {**cheap_feats, **mc_feats, "type": inst_type}

        # --- Stage 1 + Stage 2 inference (single call on pre-fitted models) ---
        X_row = pd.DataFrame([instance_features])[STAGE1_INSTANCE_FEATURES]
        t0 = time.perf_counter()
        pred_heuristic = pipeline_class.predict(X_row)[0]
        t_stage1 = time.perf_counter() - t0

        t0 = time.perf_counter()
        predict_gap(instance_features, pred_heuristic)
        t_stage2 = time.perf_counter() - t0
        t_stage1_stage2_inference = t_stage1 + t_stage2

        # --- Heuristics ---
        t_heur = {}
        heuristics_complete = True
        if n > HEURISTIC_N_CUTOFF:
            reason = f"n={n} > HEURISTIC_N_CUTOFF={HEURISTIC_N_CUTOFF}"
            skip_log.append((file, n, "all 4 heuristics", reason))
            heuristics_complete = False
            for key in HEURISTICS:
                t_heur[key] = np.nan
        else:
            for key, fn in HEURISTICS.items():
                try:
                    t0 = time.perf_counter()
                    fn(dist_matrix)
                    t_heur[key] = time.perf_counter() - t0
                except Exception as exc:  # pragma: no cover - defensive
                    reason = f"{type(exc).__name__}: {exc}"
                    skip_log.append((file, n, key, reason))
                    t_heur[key] = np.nan
                    heuristics_complete = False

        if heuristics_complete:
            t_all_heuristics = sum(t_heur.values())
            speedup = t_all_heuristics / (t_features_total + t_stage1_stage2_inference)
        else:
            t_all_heuristics = np.nan
            speedup = np.nan

        rows.append(
            {
                "file": file,
                "n": n,
                "type": inst_type,
                "t_cheap_features": t_cheap,
                "t_monte_carlo_features": t_mc,
                "t_features_total": t_features_total,
                "t_nn": t_heur.get("nn", np.nan),
                "t_greedy": t_heur.get("greedy", np.nan),
                "t_insertion": t_heur.get("insertion", np.nan),
                "t_christofides": t_heur.get("christofides", np.nan),
                "t_all_heuristics": t_all_heuristics,
                "t_stage1_stage2_inference": t_stage1_stage2_inference,
                "speedup": speedup,
            }
        )

        status = f"speedup={speedup:.3f}x" if heuristics_complete else "heuristics skipped"
        print(f"[{idx}/{len(instances)}] {file:20s} n={n:6d}  t_features={t_features_total:7.3f}s  {status}")

    results_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSaved full per-instance table to {RESULTS_CSV_PATH}\n")

    # -----------------------------------------------------------------
    # Summary: bucketed mean speedup
    # -----------------------------------------------------------------
    complete = results_df.dropna(subset=["speedup"])
    n_excluded = len(results_df) - len(complete)

    print("=" * 78)
    print("SUMMARY: mean speedup by instance size (only instances where all 4")
    print("heuristics completed, i.e. n <= HEURISTIC_N_CUTOFF = %d)" % HEURISTIC_N_CUTOFF)
    print("=" * 78)
    header = f"{'Bucket':20s} | {'n instances':>11s} | {'mean n':>8s} | {'mean speedup':>12s}"
    print(header)
    print("-" * len(header))
    bucket_means = []
    for lo, hi, bucket_name in SIZE_BUCKETS:
        bucket_df = complete[(complete["n"] >= lo) & (complete["n"] <= hi)]
        if len(bucket_df) == 0:
            print(f"{bucket_name:20s} | {'0':>11s} | {'--':>8s} | {'--':>12s}")
            continue
        mean_speedup = bucket_df["speedup"].mean()
        bucket_means.append((bucket_name, mean_speedup, len(bucket_df)))
        print(
            f"{bucket_name:20s} | {len(bucket_df):11d} | {bucket_df['n'].mean():8.1f} | "
            f"{mean_speedup:11.3f}x"
        )
    print()

    overall_mean_speedup = complete["speedup"].mean()
    largest_bucket_name, largest_bucket_speedup, largest_bucket_count = bucket_means[-1]
    mc_fraction = (complete["t_monte_carlo_features"] / complete["t_features_total"]).mean()
    n_le_cutoff = int((results_df["n"] <= HEURISTIC_N_CUTOFF).sum())

    print(
        f"Overall: mean speedup across all {len(complete)} instances with complete heuristic "
        f"timing = {overall_mean_speedup:.3f}x. For the largest size bucket "
        f"({largest_bucket_name}, n={largest_bucket_count} instances), mean speedup = "
        f"{largest_bucket_speedup:.3f}x."
    )
    print(
        f"Monte Carlo triangle-inequality features (S={200_000} samples each) account for "
        f"{mc_fraction * 100:.1f}% of total feature-extraction time on average across all "
        f"{len(results_df)} instances."
    )
    print(f"\n{n_excluded} of {len(results_df)} instances excluded from the speedup calculation "
          f"(heuristics not fully run) — see skip log below.")

    # -----------------------------------------------------------------
    # Self-review 1: skip log
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SELF-REVIEW 1: skipped instances/heuristics (auditability)")
    print("=" * 78)
    if skip_log:
        for file, n, what, reason in skip_log:
            print(f"  - {file} (n={n}): skipped {what} — {reason}")
    else:
        print("  (none)")

    # -----------------------------------------------------------------
    # Self-review 2: re-check the speedup formula
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SELF-REVIEW 2: speedup formula check")
    print("=" * 78)
    print(
        "  speedup = t_all_heuristics / (t_features_total + t_stage1_stage2_inference)\n"
        "  where t_all_heuristics = t_nn + t_greedy + t_insertion + t_christofides (SUM of all\n"
        "  four, not just the cheapest) — matches code above exactly. This reflects a user who\n"
        "  would otherwise run all four heuristics and compare their costs directly, not a user\n"
        "  who magically already knows which single heuristic is cheapest."
    )

    # -----------------------------------------------------------------
    # Self-review 3 / final markdown summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("COPYABLE MARKDOWN SUMMARY")
    print("=" * 78)
    print(f"""
## Timing benchmark summary

- **Instances loaded**: {len(instances)} (data/raw/ + data/generated/), n ranging from
  {int(results_df['n'].min())} to {int(results_df['n'].max())}.
- **Heuristic cutoff**: n <= {HEURISTIC_N_CUTOFF} ({n_le_cutoff} of {len(results_df)} instances qualify;
  see skip log for exact exclusions); insertion_tsp_from_matrix / christofides_tsp are O(n^3)
  pure Python and become impractical to run to completion above this size within one session
  (~25s/~12s each at n=800, growing cubically).
- **Instances excluded from speedup**: {n_excluded} of {len(results_df)} (heuristics not fully run — see skip log).
- **Overall mean speedup** (all-4-heuristics-cost vs. features+inference cost), across
  {len(complete)} complete instances: **{overall_mean_speedup:.3f}x**.
- **Largest-bucket mean speedup** ({largest_bucket_name}, n={largest_bucket_count}): **{largest_bucket_speedup:.3f}x**.
- **Speedup vs. size trend**: {"speedup increases with instance size" if bucket_means[-1][1] > bucket_means[0][1] else "speedup does not clearly increase with instance size"}
  (bucket means: {", ".join(f"{name}={val:.3f}x" for name, val, _ in bucket_means)}).
- **Monte Carlo cost fraction**: the two O(200,000-sample) triangle-inequality features
  (triangle_violation_rate + avg_violation_magnitude) account for **{mc_fraction * 100:.1f}%**
  of total feature-extraction time on average, across all {len(results_df)} instances.
- **Stage 1+2 inference**: timed as a SINGLE predict()+predict_gap() call per instance on one
  already-fitted RF+Quantile pipeline pair (fit once from data/Dataset.csv, SEED=1) — no
  pretrained/serialized model artifacts exist in src/models.py to load instead.
- **Caveat**: all timings are single-run wall-clock measurements (time.perf_counter()), not
  averaged over repeats; heuristic runtimes especially can vary run to run.
""")


if __name__ == "__main__":
    main()
