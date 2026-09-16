"""S2.7 / reviewer Major Issue 10: statistical significance testing across
the repeated hold-out seeds.

The manuscript currently reports descriptive win counts ("RF+Quantile achieves
the lowest MAE on 26 of 30 seeds"). That is a tally, not an inference. This
script converts those tallies into formal paired significance tests.

The design is PAIRED: every seed applies the same instance-level 80/20 split
to all four pipelines, so per-seed differences cancel the split-to-split
variation that otherwise dominates (the raw MAE standard deviation is +/-26pp,
which would swamp any unpaired comparison).

Four methodological points that the tests must respect:

  1. The 30 hold-out splits are NOT independent. All are drawn from the same
     160 instances, so the test sets overlap heavily. A naive paired t-test on
     repeated random subsampling has badly inflated Type I error
     (Dietterich 1998; Nadeau & Bengio 2003). We therefore also report the
     corrected resampled t-test, which inflates the variance estimate by
     (1/k + n_test/n_train) = (1/30 + 0.25). That corrected test is the
     defensible one whenever it disagrees with the naive test.

  2. The per-seed differences are heavily skewed (MAE ranges 2.97-82pp across
     seeds), so normality is not credible. Wilcoxon signed-rank is the PRIMARY
     test; the t-tests are confirmatory.

  3. Four pipelines give six pairwise comparisons per metric. Uncorrected,
     a false positive is expected by chance. Holm-Bonferroni is applied within
     each metric family. The PRIMARY analysis pre-specifies RF+Quantile (the
     pipeline the paper proposes) as the reference, giving 3 comparisons per
     metric; the full 6-pair matrix is reported as SECONDARY/exploratory.

  4. classifier_accuracy is identical between RF+Quantile and RF+Conformal
     (and between the two XGB variants) because the uncertainty wrapper does
     not touch Stage 1. Testing those pairs would yield all-zero differences
     and an undefined Wilcoxon statistic. Stage 1 is therefore reported as a
     single paired RF-vs-XGB comparison.

A one-sample test of coverage against the nominal 0.90 is also included: it
asks directly whether each pipeline's intervals are valid, which is what the
conformal machinery is supposed to guarantee.

Usage:
    python experiments/significance_tests.py
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
HOLDOUT_CSV = RESULTS_DIR / "multi_seed_holdout.csv"
OUT_CSV = RESULTS_DIR / "significance_tests.csv"

PIPELINES = ["RF+Quantile", "RF+Conformal", "XGB+Quantile", "XGB+Conformal"]
REFERENCE = "RF+Quantile"

NOMINAL_COVERAGE = 0.90
ALPHA = 0.05

# metric -> (column, direction) where direction is "lower" or "higher" = better
METRICS = {
    "gap_mae": ("gap_mae", "lower"),
    "r2_log": ("r2_log", "higher"),
    "median_interval_width": ("median_interval_width", "lower"),
    "coverage_error": ("_coverage_error", "lower"),
}

OUT_COLUMNS = [
    "analysis", "metric", "direction_better", "comparison", "n_seeds",
    "mean_diff", "ci_low", "ci_high",
    "wilcoxon_stat", "wilcoxon_p", "wilcoxon_n_nonzero", "wilcoxon_mode",
    "t_stat", "t_p", "nb_t_stat", "nb_p",
    "cohens_dz", "rank_biserial", "wins", "losses", "ties",
    "p_adj_holm", "significant_at_05",
]


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------
def load_and_verify() -> pd.DataFrame:
    if not HOLDOUT_CSV.exists():
        sys.exit(f"STOP: {HOLDOUT_CSV} does not exist. Run S2.5 first; this "
                 f"script will not regenerate or substitute data.")

    df = pd.read_csv(HOLDOUT_CSV)

    problems = []
    if len(df) != 120:
        problems.append(f"expected 120 rows, found {len(df)}")
    if sorted(df["pipeline"].unique()) != sorted(PIPELINES):
        problems.append(f"pipelines are {sorted(df['pipeline'].unique())}, "
                        f"expected {sorted(PIPELINES)}")
    if df["seed"].nunique() != 30:
        problems.append(f"expected 30 seeds, found {df['seed'].nunique()}")
    if df.duplicated(["seed", "pipeline"]).any():
        problems.append("duplicate (seed, pipeline) rows present")
    metric_cols = [c for c in df.columns if c not in ("seed", "pipeline")]
    if df[metric_cols].isna().any().any():
        na = df[metric_cols].isna().sum()
        problems.append(f"missing values: {na[na > 0].to_dict()}")

    if problems:
        sys.exit("STOP: results/multi_seed_holdout.csv failed precondition:\n  - "
                 + "\n  - ".join(problems))

    print(f"Precondition OK: {len(df)} rows = {df['seed'].nunique()} seeds "
          f"x {df['pipeline'].nunique()} pipelines, no missing values.\n")
    return df


def verify_split_sizes() -> tuple[int, int, float]:
    """Read the true per-seed train/test instance counts out of the pipeline
    code rather than assuming 128/32."""
    from src.config import DATASET_PATH
    from src.models import prepare_stage2_split

    df = pd.read_csv(DATASET_PATH)
    sizes = set()
    for seed in range(1, 31):
        sp = prepare_stage2_split(df, seed=seed)
        sizes.add((len(sp["train_files"]), len(sp["test_files"])))

    if len(sizes) != 1:
        sys.exit(f"STOP: split sizes vary across seeds: {sizes}. The "
                 f"Nadeau-Bengio correction assumes a constant split ratio.")

    n_train, n_test = sizes.pop()
    rho = n_test / n_train
    print(f"Split sizes verified from src/models.py: n_train={n_train} "
          f"instances, n_test={n_test} instances, rho = n_test/n_train = {rho:.4f}")
    if (n_train, n_test) != (128, 32):
        print(f"  NOTE: this differs from the assumed 128/32; using the true "
              f"values above.")
    else:
        print(f"  (matches the assumed 128/32)")
    print()
    return n_train, n_test, rho


# ---------------------------------------------------------------------------
# Test machinery
# ---------------------------------------------------------------------------
def nadeau_bengio_ttest(diff: np.ndarray, rho: float) -> tuple[float, float]:
    """Corrected resampled t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004).

    The naive paired t-test uses var/k. Because the k resampled test sets
    overlap, that understates the variance. The corrected test inflates it to

        var * (1/k + n_test/n_train)

    NOTE ON THE CONSTANT. Two equivalent conventions appear in the literature
    and it is easy to conflate them:

        rho defined as n_test/n_train        -> inflation (1/k + rho)
        rho defined as n_test/(n_train+n_test) -> inflation (1/k + rho/(1-rho))

    Here n_test=32, n_train=128, n_total=160, so the first convention gives
    rho=0.25 and the second gives rho=0.20; both yield the SAME inflation
    term of 0.25. Applying rho/(1-rho) to rho=0.25 would give 0.333, which
    double-applies the transformation and overcorrects. `rho` is passed in as
    n_test/n_train, so the correct term is (1/k + rho).
    """
    k = len(diff)
    mean = float(np.mean(diff))
    var = float(np.var(diff, ddof=1))
    if var == 0.0:
        return float("nan"), 1.0
    corrected_se = np.sqrt(var * (1.0 / k + rho))
    t = mean / corrected_se
    p = 2.0 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def paired_tests(a: np.ndarray, b: np.ndarray, rho: float) -> dict:
    """Paired comparison of a (reference) against b. diff = b - a."""
    diff = b - a
    k = len(diff)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))

    # 95% CI on the mean paired difference (naive, uncorrected)
    se = sd_diff / np.sqrt(k)
    tcrit = stats.t.ppf(0.975, df=k - 1)
    ci_low, ci_high = mean_diff - tcrit * se, mean_diff + tcrit * se

    # Wilcoxon signed-rank (PRIMARY). Zeros are dropped ("wilcox" convention);
    # an exact test is used when there are no ties and n is small enough.
    nonzero = diff[diff != 0]
    n_nonzero = int(len(nonzero))
    has_ties = len(np.unique(np.abs(nonzero))) < n_nonzero
    if n_nonzero == 0:
        w_stat, w_p, mode = float("nan"), 1.0, "undefined (all differences zero)"
    else:
        mode = "exact" if (n_nonzero <= 25 and not has_ties) else "approx (normal)"
        try:
            w_stat, w_p = stats.wilcoxon(
                diff, zero_method="wilcox",
                mode="exact" if mode == "exact" else "approx",
            )
        except TypeError:  # scipy >= 1.9 renamed mode -> method
            w_stat, w_p = stats.wilcoxon(
                diff, zero_method="wilcox",
                method="exact" if mode == "exact" else "approx",
            )
        w_stat, w_p = float(w_stat), float(w_p)

    # rank-biserial correlation for the signed-rank test
    if n_nonzero > 0:
        ranks = stats.rankdata(np.abs(nonzero))
        r_plus = float(ranks[nonzero > 0].sum())
        r_minus = float(ranks[nonzero < 0].sum())
        total = r_plus + r_minus
        rank_biserial = (r_plus - r_minus) / total if total > 0 else float("nan")
    else:
        rank_biserial = float("nan")

    # paired t-test (secondary) and Nadeau-Bengio corrected version
    if sd_diff == 0.0:
        t_stat, t_p = float("nan"), 1.0
    else:
        t_stat, t_p = stats.ttest_rel(b, a)
        t_stat, t_p = float(t_stat), float(t_p)
    nb_t, nb_p = nadeau_bengio_ttest(diff, rho)

    cohens_dz = mean_diff / sd_diff if sd_diff > 0 else float("nan")

    return {
        "n_seeds": k,
        "mean_diff": mean_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "wilcoxon_n_nonzero": n_nonzero,
        "wilcoxon_mode": mode,
        "t_stat": t_stat,
        "t_p": t_p,
        "nb_t_stat": nb_t,
        "nb_p": nb_p,
        "cohens_dz": float(cohens_dz),
        "rank_biserial": float(rank_biserial),
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone-enforced)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """seed x pipeline wide frame, with the derived coverage-error column."""
    df = df.copy()
    df["_coverage_error"] = (df["coverage"] - NOMINAL_COVERAGE).abs()
    return df


def run_pairwise(df: pd.DataFrame, rho: float) -> pd.DataFrame:
    rows = []

    for metric_name, (col, direction) in METRICS.items():
        wide = df.pivot(index="seed", columns="pipeline", values=col).sort_index()

        # --- PRIMARY: RF+Quantile vs each of the other three -------------
        primary_rows = []
        for other in [p for p in PIPELINES if p != REFERENCE]:
            a = wide[REFERENCE].to_numpy(float)
            b = wide[other].to_numpy(float)
            # orient so that "wins" means the reference is better
            if direction == "lower":
                res = paired_tests(a, b, rho)      # diff = other - ref; >0 => ref better
            else:
                res = paired_tests(b, a, rho)      # diff = ref - other; >0 => ref better
            res.update({
                "analysis": "primary",
                "metric": metric_name,
                "direction_better": direction,
                "comparison": f"{REFERENCE} vs {other}",
            })
            primary_rows.append(res)

        adj = holm_bonferroni([r["nb_p"] for r in primary_rows])
        for r, a_ in zip(primary_rows, adj):
            r["p_adj_holm"] = a_
            r["significant_at_05"] = bool(a_ < ALPHA)
        rows.extend(primary_rows)

        # --- SECONDARY: full 6-pair exploratory matrix -------------------
        sec_rows = []
        for p1, p2 in combinations(PIPELINES, 2):
            a = wide[p1].to_numpy(float)
            b = wide[p2].to_numpy(float)
            if direction == "lower":
                res = paired_tests(a, b, rho)
            else:
                res = paired_tests(b, a, rho)
            res.update({
                "analysis": "secondary",
                "metric": metric_name,
                "direction_better": direction,
                "comparison": f"{p1} vs {p2}",
            })
            sec_rows.append(res)
        adj = holm_bonferroni([r["nb_p"] for r in sec_rows])
        for r, a_ in zip(sec_rows, adj):
            r["p_adj_holm"] = a_
            r["significant_at_05"] = bool(a_ < ALPHA)
        rows.extend(sec_rows)

    return pd.DataFrame(rows)


def run_stage1(df: pd.DataFrame, rho: float) -> pd.DataFrame:
    """Stage 1 is a single RF-vs-XGB comparison, not six."""
    wide = df.pivot(index="seed", columns="pipeline",
                    values="classifier_accuracy").sort_index()

    assert (wide["RF+Quantile"] == wide["RF+Conformal"]).all(), \
        "RF+Quantile and RF+Conformal should share an identical Stage 1 model"
    assert (wide["XGB+Quantile"] == wide["XGB+Conformal"]).all(), \
        "XGB+Quantile and XGB+Conformal should share an identical Stage 1 model"
    print("Stage 1 degeneracy verified: classifier_accuracy is identical within "
          "each backbone.\n  -> reporting ONE paired RF-vs-XGB test, not six.\n")

    rf = wide["RF+Quantile"].to_numpy(float)
    xgb = wide["XGB+Quantile"].to_numpy(float)
    res = paired_tests(xgb, rf, rho)   # diff = rf - xgb; >0 => RF better
    res.update({
        "analysis": "primary",
        "metric": "classifier_accuracy",
        "direction_better": "higher",
        "comparison": "RF (Stage 1) vs XGB (Stage 1)",
        "p_adj_holm": res["nb_p"],          # family of one
        "significant_at_05": bool(res["nb_p"] < ALPHA),
    })
    return pd.DataFrame([res])


def run_coverage_validity(df: pd.DataFrame, rho: float) -> pd.DataFrame:
    """One-sample Wilcoxon of per-seed coverage against nominal 0.90."""
    wide = df.pivot(index="seed", columns="pipeline", values="coverage").sort_index()
    rows = []
    for p in PIPELINES:
        cov = wide[p].to_numpy(float)
        nominal = np.full_like(cov, NOMINAL_COVERAGE)
        res = paired_tests(nominal, cov, rho)   # diff = cov - nominal
        res.update({
            "analysis": "coverage_validity",
            "metric": "coverage_vs_nominal",
            "direction_better": "closer to 0.90",
            "comparison": f"{p} vs nominal 90%",
        })
        rows.append(res)
    adj = holm_bonferroni([r["nb_p"] for r in rows])
    for r, a_ in zip(rows, adj):
        r["p_adj_holm"] = a_
        r["significant_at_05"] = bool(a_ < ALPHA)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "  n/a  "
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_block(title: str, sub: pd.DataFrame) -> None:
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)
    hdr = (f"{'Comparison':34s} | {'mean diff':>10s} | {'95% CI':>20s} | "
           f"{'Wilcox p':>9s} | {'t p':>9s} | {'NB p':>9s} | {'Holm':>9s} | "
           f"{'d_z':>6s} | {'wins':>6s} | sig")
    print(hdr)
    print("-" * len(hdr))
    for _, r in sub.iterrows():
        ci = f"[{r['ci_low']:7.3f},{r['ci_high']:7.3f}]"
        wins = f"{int(r['wins'])}/{int(r['n_seeds'])}"
        print(f"{r['comparison']:34s} | {r['mean_diff']:10.4f} | {ci:>20s} | "
              f"{fmt_p(r['wilcoxon_p']):>9s} | {fmt_p(r['t_p']):>9s} | "
              f"{fmt_p(r['nb_p']):>9s} | {fmt_p(r['p_adj_holm']):>9s} | "
              f"{r['cohens_dz']:6.3f} | {wins:>6s} | "
              f"{'YES' if r['significant_at_05'] else 'no'}")


def main() -> None:
    df = load_and_verify()
    n_train, n_test, rho = verify_split_sizes()
    df = build_matrix(df)

    stage1 = run_stage1(df, rho)
    pairwise = run_pairwise(df, rho)
    coverage = run_coverage_validity(df, rho)

    all_rows = pd.concat([stage1, pairwise, coverage], ignore_index=True)
    all_rows = all_rows[OUT_COLUMNS]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(OUT_CSV, index=False)

    print("Metric directions: gap_mae lower=better; r2_log higher=better; "
          "median_interval_width lower=better;")
    print("                   coverage_error = |coverage - 0.90|, lower=better.")
    print("All paired comparisons are oriented so that a POSITIVE mean_diff and "
          "a HIGH win count")
    print("mean the FIRST-named pipeline is better.")
    print(f"\nNadeau-Bengio correction: k={len(df['seed'].unique())} seeds, "
          f"rho=n_test/n_train={rho:.4f}, variance inflation factor "
          f"(1/k + rho) = {1/30 + rho:.4f} vs naive 1/k = {1/30:.4f} "
          f"(SE ratio {np.sqrt((1/30 + rho)/(1/30)):.2f}x).")
    print("Holm-Bonferroni is applied to the Nadeau-Bengio p-values within each "
          "metric family.")

    print_block("STAGE 1 (single pre-specified comparison)", stage1)

    for metric_name in METRICS:
        sub = pairwise[(pairwise["metric"] == metric_name) &
                       (pairwise["analysis"] == "primary")]
        print_block(f"PRIMARY -- {metric_name} "
                    f"({METRICS[metric_name][1]}=better), Holm over 3 comparisons",
                    sub)

    for metric_name in METRICS:
        sub = pairwise[(pairwise["metric"] == metric_name) &
                       (pairwise["analysis"] == "secondary")]
        print_block(f"SECONDARY/EXPLORATORY -- {metric_name}, "
                    f"Holm over all 6 pairs", sub)

    print_block("COVERAGE VALIDITY: per-seed coverage vs nominal 90% "
                "(positive mean diff = over-coverage)", coverage)

    # ------------------------------------------------------------------
    # Disagreement audit between naive and corrected t-tests
    # ------------------------------------------------------------------
    print("\n" + "=" * 108)
    print("NAIVE vs CORRECTED t-TEST DISAGREEMENTS (raw p, uncorrected for "
          "multiplicity)")
    print("=" * 108)
    dis = all_rows[(all_rows["t_p"] < ALPHA) & (all_rows["nb_p"] >= ALPHA)]
    if len(dis) == 0:
        print("None: no comparison flips from significant to non-significant "
              "under the Nadeau-Bengio correction.")
    else:
        print(f"{len(dis)} comparison(s) are significant under the NAIVE paired "
              f"t-test but NOT under the")
        print("Nadeau-Bengio corrected test. The corrected test is the "
              "defensible one, because the 30")
        print(f"hold-out splits share the same {n_train + n_test} instances and "
              f"their test sets overlap heavily.")
        for _, r in dis.iterrows():
            print(f"  - [{r['analysis']:>17s}] {r['metric']:22s} "
                  f"{r['comparison']:34s} naive p={fmt_p(r['t_p'])} -> "
                  f"NB p={fmt_p(r['nb_p'])}")

    # ------------------------------------------------------------------
    # Power analysis: what could this design have detected at all?
    # ------------------------------------------------------------------
    k = 30
    infl = 1.0 / k + rho
    tcrit_raw = stats.t.ppf(0.975, k - 1)
    tcrit_holm3 = stats.t.ppf(1 - (ALPHA / 3) / 2, k - 1)
    dz_raw = tcrit_raw * np.sqrt(infl)
    dz_holm = tcrit_holm3 * np.sqrt(infl)

    print("\n" + "=" * 108)
    print("POWER ANALYSIS OF THE CORRECTED TEST")
    print("=" * 108)
    print(f"Under the Nadeau-Bengio test, t = d_z / sqrt(1/k + rho) = d_z / "
          f"{np.sqrt(infl):.4f}. With k={k} and rho={rho}:")
    print(f"  smallest detectable |d_z| at raw alpha=.05          : {dz_raw:.3f}")
    print(f"  smallest detectable |d_z| after Holm over 3 tests   : {dz_holm:.3f}")
    print()
    print("CRITICAL: adding more seeds cannot fix this. The inflation term "
          "(1/k + rho) is bounded")
    print(f"below by rho = {rho}, so as k -> infinity the smallest detectable "
          f"|d_z| only falls to")
    print(f"{stats.norm.ppf(0.975) * np.sqrt(rho):.3f} (raw). Going from k=30 to "
          f"k=infinity buys just "
          f"{(np.sqrt(infl / rho) - 1) * 100:.1f}% in the t-statistic.")
    print("The binding constraint is the 160-instance dataset, NOT the number "
          "of resamples.")
    print()
    pr = all_rows[(all_rows["analysis"] == "primary") &
                  (all_rows["metric"] != "classifier_accuracy")]
    n_under = int((pr["cohens_dz"].abs() <= dz_holm).sum())
    print(f"{n_under} of {len(pr)} primary comparisons have |d_z| below the "
          f"Holm-detectable floor of {dz_holm:.3f}")
    print("and are therefore UNDERPOWERED by construction: a null result for "
          "them is uninformative,")
    print("not evidence of equivalence.")

    print("\n" + "=" * 108)
    print("CAVEAT ON THE WILCOXON p-VALUES")
    print("=" * 108)
    print("The Wilcoxon signed-rank test assumes the 30 per-seed differences "
          "are independent. They")
    print("are NOT: the splits are resampled from the same 160 instances. "
          "There is no standard")
    print("dependence correction for the signed-rank test, so its p-values "
          "here are ANTICONSERVATIVE")
    print("(too small) by roughly the same factor that separates the naive "
          "and corrected t-tests.")
    print("They are reported because they were pre-specified as primary and "
          "because they are robust")
    print("to the heavy skew, but claims in the manuscript should rest on the "
          "Holm-adjusted")
    print("Nadeau-Bengio column, which is the only one that accounts for both "
          "dependence and")
    print("multiplicity.")

    print(f"\nSaved {len(all_rows)} test rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
