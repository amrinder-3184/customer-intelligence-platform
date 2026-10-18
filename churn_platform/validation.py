"""Schema and data-quality validation for the raw Telco churn CSV.

Design:
- The CSV is loaded with every column as *text* (dtype=str). That way pandas
  never silently converts or drops anything; every type conversion happens
  here, where a failure is reported instead of hidden.
- validate() never modifies the data. It only returns a list of Issues.
  Cleaning (e.g. what to do with blank TotalCharges) is a modelling decision
  and belongs in the preprocessing phase.
- Each Issue is either an ERROR (the file cannot be trusted: wrong columns,
  unknown categories, duplicate IDs...) or a WARNING (a known, explainable
  quirk that later phases must handle deliberately).

The expected schema below was written AFTER inspecting the real file
(IBM Telco Customer Churn, 7043 rows x 21 columns), not guessed.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

YES_NO = {"Yes", "No"}
INTERNET_ADDON = {"Yes", "No", "No internet service"}

# Categorical columns -> the complete set of allowed values.
CATEGORICAL_COLUMNS: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {"0", "1"},  # stored as 0/1, unlike the other Yes/No flags
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": YES_NO,
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    TARGET_COLUMN: YES_NO,
}

# Numeric columns -> (minimum, maximum) that make business sense.
# These are sanity bounds, not the observed min/max.
NUMERIC_COLUMNS: dict[str, tuple[float, float]] = {
    "tenure": (0, 120),  # months
    "MonthlyCharges": (0, 1_000),
    "TotalCharges": (0, 100_000),
}

EXPECTED_COLUMNS = [ID_COLUMN, *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]

INTERNET_ADDON_COLUMNS = [c for c, allowed in CATEGORICAL_COLUMNS.items() if allowed == INTERNET_ADDON]


@dataclass
class Issue:
    severity: str  # "ERROR" or "WARNING"
    check: str
    message: str
    row_count: int = 0
    examples: list = field(default_factory=list)  # a few customerIDs, for debugging


def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV with every value as a string and no automatic NaN parsing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m churn_platform.download"
        )
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _ids(df: pd.DataFrame, mask: pd.Series, limit: int = 5) -> list:
    if ID_COLUMN not in df.columns:
        return []
    return df.loc[mask, ID_COLUMN].head(limit).tolist()


def validate(df: pd.DataFrame) -> list[Issue]:
    """Run all checks and return the issues found (empty list = clean)."""
    issues: list[Issue] = []

    # 1. Column set. If columns are missing, later checks would crash, so stop.
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        issues.append(Issue("ERROR", "missing_columns", f"Missing columns: {missing}"))
        return issues
    if extra:
        issues.append(Issue("WARNING", "extra_columns", f"Unexpected columns (ignored): {extra}"))

    if df.empty:
        issues.append(Issue("ERROR", "empty", "Dataset has no rows"))
        return issues

    # 2. Customer ID: present and unique. Duplicate IDs could put the same
    #    customer in both train and test sets later (leakage).
    blank_id = df[ID_COLUMN].str.strip() == ""
    if blank_id.any():
        issues.append(Issue("ERROR", "blank_id", "Blank customerID", int(blank_id.sum())))
    dup_id = df[ID_COLUMN].duplicated(keep=False) & ~blank_id
    if dup_id.any():
        issues.append(
            Issue("ERROR", "duplicate_id", "customerID appears more than once",
                  int(dup_id.sum()), _ids(df, dup_id))
        )

    # 3. Categorical values must come from the known set.
    for col, allowed in CATEGORICAL_COLUMNS.items():
        bad = ~df[col].isin(allowed)
        if bad.any():
            found = sorted(df.loc[bad, col].unique().tolist())[:5]
            issues.append(
                Issue("ERROR", "invalid_category",
                      f"{col}: values outside {sorted(allowed)}; found {found}",
                      int(bad.sum()), _ids(df, bad))
            )

    # 4. Numeric columns: parseable and within sane bounds.
    numeric = {}
    for col, (low, high) in NUMERIC_COLUMNS.items():
        blank = df[col].str.strip() == ""
        values = pd.to_numeric(df[col], errors="coerce")
        numeric[col] = values
        unparseable = values.isna() & ~blank
        if unparseable.any():
            issues.append(
                Issue("ERROR", "not_numeric", f"{col}: non-numeric values",
                      int(unparseable.sum()), _ids(df, unparseable))
            )
        if blank.any():
            # TotalCharges is blank only for brand-new customers (tenure 0):
            # they have not been billed yet. Explainable, so a WARNING.
            explained = col == "TotalCharges" and (numeric["tenure"][blank] == 0).all()
            issues.append(
                Issue("WARNING" if explained else "ERROR", "blank_numeric",
                      f"{col}: blank values" + (" (all have tenure == 0)" if explained else ""),
                      int(blank.sum()), _ids(df, blank))
            )
        out_of_range = (values < low) | (values > high)
        if out_of_range.any():
            issues.append(
                Issue("ERROR", "out_of_range", f"{col}: outside [{low}, {high}]",
                      int(out_of_range.sum()), _ids(df, out_of_range))
            )

    # 5. Cross-column consistency rules.
    no_phone = df["PhoneService"] == "No"
    inconsistent = no_phone != (df["MultipleLines"] == "No phone service")
    if inconsistent.any():
        issues.append(
            Issue("ERROR", "inconsistent_phone",
                  "PhoneService and MultipleLines disagree",
                  int(inconsistent.sum()), _ids(df, inconsistent))
        )
    no_internet = df["InternetService"] == "No"
    for col in INTERNET_ADDON_COLUMNS:
        inconsistent = no_internet != (df[col] == "No internet service")
        if inconsistent.any():
            issues.append(
                Issue("ERROR", "inconsistent_internet",
                      f"InternetService and {col} disagree",
                      int(inconsistent.sum()), _ids(df, inconsistent))
            )

    # 6. Identical feature rows under different IDs. Not an error (two
    #    customers can have the same plan), but worth knowing about.
    feature_cols = [c for c in EXPECTED_COLUMNS if c not in (ID_COLUMN, TARGET_COLUMN)]
    dup_features = df.duplicated(subset=feature_cols, keep=False)
    if dup_features.any():
        issues.append(
            Issue("WARNING", "duplicate_feature_rows",
                  "Different customers with identical feature values",
                  int(dup_features.sum()), _ids(df, dup_features))
        )

    return issues


def summarize(df: pd.DataFrame) -> dict:
    """Basic facts about the dataset, for the inspection report."""
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
    }
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts()
        summary["target_counts"] = {k: int(v) for k, v in counts.items()}
        summary["churn_rate"] = round(float((df[TARGET_COLUMN] == "Yes").mean()), 4)
    numeric_stats = {}
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            numeric_stats[col] = {
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
                "missing_or_blank": int(values.isna().sum()),
            }
    summary["numeric"] = numeric_stats
    summary["categorical_counts"] = {
        col: {k: int(v) for k, v in df[col].value_counts().items()}
        for col in CATEGORICAL_COLUMNS
        if col in df.columns and col != TARGET_COLUMN
    }
    return summary
