"""One-off migration: renormalize ``min_edge_weight`` and
``avg_violation_magnitude`` in ``data/Dataset.csv`` for the S2.1 scale-
invariance fix (see src/features.py).

Before this fix, ``extract_all_features()`` returned ``min_edge_weight`` and
``avg_violation_magnitude`` in raw distance units. ``data/Dataset.csv`` was
built with the old, unnormalized formula, so its columns are now stale and
inconsistent with the fixed feature-extraction code in src/features.py.

This script recomputes both columns in place as
``old_value / mean_edge_weight``, using the ``mean_edge_weight`` column
already present in the CSV (no need to re-run the heuristics or re-read the
raw .tsp files). The original file is backed up first.

Usage:
    python experiments/migrate_dataset_normalization.py

If ``mean_edge_weight`` is not present as a column in data/Dataset.csv, this
script refuses to guess and prints an error instead — in that case, Dataset.csv
must be regenerated from raw .tsp files via build_dataset.py.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASET_PATH  # noqa: E402

BACKUP_PATH = DATASET_PATH.parent / "Dataset.csv.pre_normalization_backup"


def migrate(dataset_path: Path = DATASET_PATH, backup_path: Path = BACKUP_PATH) -> None:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found. Nothing to migrate.")
        sys.exit(1)

    df = pd.read_csv(dataset_path)

    required_cols = {"min_edge_weight", "avg_violation_magnitude", "mean_edge_weight"}
    missing = required_cols - set(df.columns)
    if missing:
        print(
            f"ERROR: {dataset_path} is missing column(s) {sorted(missing)}.\n"
            "Cannot safely renormalize min_edge_weight / avg_violation_magnitude "
            "without the mean_edge_weight column already present in the CSV.\n"
            "Regenerate data/Dataset.csv from raw .tsp files instead, via:\n"
            "    python -m src.build_dataset"
        )
        sys.exit(1)

    if (df["mean_edge_weight"] <= 0).any():
        print(
            "ERROR: found non-positive mean_edge_weight value(s) in "
            f"{dataset_path} — cannot safely divide. Regenerate the dataset "
            "via build_dataset.py instead."
        )
        sys.exit(1)

    print(f"Backing up original dataset to {backup_path} ...")
    shutil.copy2(dataset_path, backup_path)

    old_min = df["min_edge_weight"].copy()
    old_avg_violation = df["avg_violation_magnitude"].copy()

    df["min_edge_weight"] = old_min / df["mean_edge_weight"]
    df["avg_violation_magnitude"] = old_avg_violation / df["mean_edge_weight"]

    df.to_csv(dataset_path, index=False)

    print(f"Migrated {len(df)} rows in {dataset_path}.")
    print(
        "min_edge_weight: mean before={:.4f}, mean after={:.4f}".format(
            old_min.mean(), df["min_edge_weight"].mean()
        )
    )
    print(
        "avg_violation_magnitude: mean before={:.4f}, mean after={:.4f}".format(
            old_avg_violation.mean(), df["avg_violation_magnitude"].mean()
        )
    )
    print(f"Original (pre-migration) file preserved at {backup_path}.")


if __name__ == "__main__":
    migrate()
