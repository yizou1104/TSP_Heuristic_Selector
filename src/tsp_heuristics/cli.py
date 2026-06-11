"""Command-line entry point: tsp-predict."""

from __future__ import annotations

import json
import sys
import time

import click

from . import __version__
from .features import extract_features
from .models import predict
from .parsers import load_distance_matrix


_MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "lr": "Logistic Regression"}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version")
@click.argument("tsp_file", type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    "--model",
    default="rf",
    type=click.Choice(["rf", "xgb", "lr"]),
    show_default=True,
    help="ML model backend.",
)
@click.option(
    "--seed",
    default=1,
    type=click.IntRange(1, 3),
    show_default=True,
    help="Model seed 1–3 (rf/xgb only; ignored for lr).",
)
@click.option(
    "--no-intervals",
    is_flag=True,
    default=False,
    help="Suppress 95%% prediction interval on the gap estimate.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Print machine-readable JSON instead of human-readable output.",
)
def predict_cli(
    tsp_file: str,
    model: str,
    seed: int,
    no_intervals: bool,
    json_output: bool,
) -> None:
    """Predict the best TSP construction heuristic and optimality gap.

    TSP_FILE: path to a TSPLIB-format .tsp file.

    \b
    Examples:
      tsp-predict att48.tsp
      tsp-predict att48.tsp --model xgb --seed 2
      tsp-predict att48.tsp --no-intervals
      tsp-predict att48.tsp --json
    """
    try:
        t0 = time.perf_counter()
        dist, meta = load_distance_matrix(tsp_file)
        feats = extract_features(dist)
        result = predict(
            feats,
            edge_type=meta["edge_type"],
            model_type=model,
            seed=seed,
            intervals=not no_intervals,
        )
        elapsed = time.perf_counter() - t0
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if json_output:
        out: dict = {
            "instance": meta["name"],
            "n_nodes": meta["n"],
            "edge_type": meta["edge_type"],
            "predicted_heuristic": result["heuristic"],
            "heuristic_probabilities": result["heuristic_proba"],
            "predicted_gap_pct": result["gap_pct"],
            "gap_lower_95_pct": result["gap_lower_95"],
            "gap_upper_95_pct": result["gap_upper_95"],
            "model": model,
            "seed": seed if model != "lr" else None,
            "elapsed_s": round(elapsed, 3),
        }
        click.echo(json.dumps(out, indent=2))
        return

    # Human-readable output
    heur = result["heuristic"]
    gap = result["gap_pct"]
    lower = result["gap_lower_95"]
    upper = result["gap_upper_95"]

    click.echo(f"Instance:        {meta['name']}  ({meta['n']} nodes, {meta['edge_type']})")
    click.echo(f"Best heuristic:  {heur}")

    if lower is not None and upper is not None:
        click.echo(f"Predicted gap:   {gap:.1f}%  [95% CI: {lower:.1f}% – {upper:.1f}%]")
    else:
        click.echo(f"Predicted gap:   {gap:.1f}%")

    seed_str = f" (seed={seed})" if model != "lr" else ""
    click.echo(f"Model used:      {_MODEL_LABELS[model]}{seed_str}")

    # Show heuristic probabilities
    proba = result["heuristic_proba"]
    if proba and len(proba) > 1:
        proba_str = "  ".join(f"{h}: {p:.2f}" for h, p in sorted(proba.items()))
        click.echo(f"Probabilities:   {proba_str}")
