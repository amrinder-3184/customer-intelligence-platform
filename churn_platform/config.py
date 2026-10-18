"""Project-wide paths and constants.

Every other module imports paths from here, so the project works no matter
which directory a command is run from.
"""

from pathlib import Path

# Repository root = the folder that contains this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"

# Phase 2 (EDA + SQL) outputs. The database is derived from the raw CSV and
# rebuilt on every run, so it is git-ignored.
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANALYSIS_DB_FILE = PROCESSED_DATA_DIR / "telco_analysis.db"
# Phase 3 train/test split (customerID -> train/test). train_baseline.py keeps
# its own identical constant (left unchanged on purpose); a test checks they agree.
SPLIT_FILE = PROCESSED_DATA_DIR / "train_test_split.csv"
SPLIT_FINGERPRINT_FILE = PROJECT_ROOT / "reports" / "split_fingerprint.json"
HISTORICAL_HASHES_FILE = PROJECT_ROOT / "reports" / "historical_results.sha256"
SQL_DIR = PROJECT_ROOT / "sql"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Phase 5: serving artifact (git-ignored, rebuilt by `python -m churn_platform.artifact`).
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_FILE = MODELS_DIR / "churn_logreg_v1.joblib"

# Public copy published by IBM (repository licence: Apache-2.0).
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

RANDOM_SEED = 42
