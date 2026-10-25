"""Project configuration for the sweet.tv TV program recommender (Kaggle).

All paths are derived from the location of this file so that scripts work
regardless of the current working directory.
"""

import os
from pathlib import Path

# Directory containing this project (midterm/).
BASE_DIR = Path(__file__).resolve().parent.parent

# Default location of the raw data (outside the repository).
_DEFAULT_DATA_DIR = r"C:\Users\User\Downloads\sweettv-tv-program-recommender"

# Override the data directory via the SWEETTV_DATA_DIR environment variable.
DATA_DIR = Path(os.environ.get("SWEETTV_DATA_DIR", _DEFAULT_DATA_DIR))

# Input files.
LOG_FILE = DATA_DIR / "dataset11-30.csv"
GRID_11_20_FILE = DATA_DIR / "export_arh_11-20-final.csv"
GRID_21_30_FILE = DATA_DIR / "export_arh_21-30-final.csv"
GRID_31_42_FILE = DATA_DIR / "export_arh_31-42-final.csv"
SUBMISSION_FILE = DATA_DIR / "submission.csv"

# Local cache / output directories.
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Reproducibility and evaluation settings.
SEED = 42
K = 5
