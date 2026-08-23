"""Central path configuration for the TSP Heuristic Selector pipeline.

Every other module in this package imports its file-system paths from here —
no module should ever hardcode a literal path (and in particular, none of the
old Colab ``/content/...`` paths from the original notebook should appear
anywhere else in this codebase).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Adjust this if your local folder name/location differs.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/yizou/projects/TSP_Heuristic_Selector")

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
RAW_EUCLIDEAN_2D_DIR = RAW_DIR / "euclidean_2d"
RAW_EXPLICIT_DIR = RAW_DIR / "explicit"
RAW_GEO_DIR = RAW_DIR / "geo"

GENERATED_DIR = DATA_DIR / "generated"
GENERATED_RANDOM_DIR = GENERATED_DIR / "random"
GENERATED_CLUSTERED_DIR = GENERATED_DIR / "clustered"
GENERATED_CHRISTOFIDES_HARD_DIR = GENERATED_DIR / "christofides_hard"

SOLUTIONS_DIR = DATA_DIR / "solutions"
SOLUTIONS_TXT_PATH = SOLUTIONS_DIR / "solutions.txt"

DATASET_PATH = DATA_DIR / "dataset.csv"

# ---------------------------------------------------------------------------
# External optimal-tour solvers.
#
# These binaries are NOT bundled with this repository — install them locally
# and place (or symlink) them at the paths below, or edit the paths to point
# at wherever you installed them:
#
#   Concorde: https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm
#   LKH-3.0.9: http://webhotel4.ruc.dk/~keld/research/LKH-3/
# ---------------------------------------------------------------------------
TOOLS_DIR = PROJECT_ROOT / "tools"
CONCORDE_PATH = TOOLS_DIR / "concorde" / "TSP" / "concorde"
LKH_PATH = TOOLS_DIR / "LKH-3.0.9" / "LKH"
