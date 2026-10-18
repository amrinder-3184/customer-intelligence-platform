"""Descriptive analysis of the Telco churn data (pure pandas, no plotting).

Everything here is DESCRIPTIVE: it measures how churn rates differ between
groups of customers that already exist. It does not say that changing a
customer's group would change their churn (that would be a causal claim the
data cannot support).

The raw DataFrame from validation.load_raw() is never modified; every
function works on a copy.
"""

import numpy as np
import pandas as pd

from churn_platform.validation import (
    CATEGORICAL_COLUMNS,
    EXPECTED_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

# Tenure groups in months. Upper bounds are inclusive. The SQL view in
# sql/create_analysis_view.sql implements the same rules independently, and
# a test checks that the two agree.
TENURE_GROUPS = ["0-12", "13-24", "25-48", "49+"]
_TENURE_EDGES = [-np.inf, 12, 24, 48, np.inf]

# Monthly-charge bands in dollars. Lower bound inclusive, upper exclusive.
# Round numbers chosen for readability; they are not optimised in any way.
CHARGE_BANDS = ["<35", "35-70", "70-90", "90+"]
_CHARGE_EDGES = [-np.inf, 35, 70, 90, np.inf]

# Columns that describe the customer. customerID is an identifier, not
# behaviour, and Churn is the outcome, so both are excluded.
FEATURE_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in (ID_COLUMN, TARGET_COLUMN)]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL_COLUMNS if c != TARGET_COLUMN]


def to_analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed copy of the raw data for analysis. The raw frame is untouched.

    Transformations (all documented in the README):
    - tenure -> int, MonthlyCharges -> float.
    - TotalCharges -> float. The 11 blank strings (" ") become NaN, NOT 0,
      so missingness stays visible. A boolean column TotalCharges_blank
      records exactly which rows were blank.
    - churned -> 1 if Churn == "Yes" else 0 (the original Churn column is kept).
    - tenure_group and monthly_charges_band are added for segment analysis.
    """
    df = raw.copy()
    df["tenure"] = pd.to_numeric(df["tenure"], errors="raise").astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="raise").astype(float)

    blank = df["TotalCharges"].str.strip() == ""
    # errors="raise": a value that is neither blank nor a number should
    # stop the analysis, not quietly become NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].mask(blank), errors="raise").astype(float)
    df["TotalCharges_blank"] = blank

    df["churned"] = (df[TARGET_COLUMN] == "Yes").astype(int)
    df["tenure_group"] = pd.cut(df["tenure"], _TENURE_EDGES, labels=TENURE_GROUPS).astype(str)
    df["monthly_charges_band"] = pd.cut(
        df["MonthlyCharges"], _CHARGE_EDGES, labels=CHARGE_BANDS, right=False
    ).astype(str)
    return df


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion.

    Better than the textbook p +/- 1.96*sqrt(p(1-p)/n) for small groups or
    rates near 0 or 1, because it never goes below 0 or above 1.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # Clip: floating-point rounding can give e.g. -2.8e-17 when successes == 0.
    return (max(0.0, float(centre - half)), min(1.0, float(centre + half)))


def churn_rate_by(df: pd.DataFrame, columns, order: list | None = None) -> pd.DataFrame:
    """Customers, churners, churn rate and 95% CI for each group.

    columns: one column name or a list of names (for cross-segments).
    order:   optional display order for a single column; otherwise rows are
             sorted by churn rate, highest first. Groups with no customers
             are left out (as in SQL GROUP BY), not shown as zero rows.
    """
    columns = [columns] if isinstance(columns, str) else list(columns)
    table = (
        df.groupby(columns, observed=True)["churned"]
        .agg(customers="count", churned="sum")
        .reset_index()
    )
    table["churn_rate"] = table["churned"] / table["customers"]
    intervals = [wilson_interval(k, n) for k, n in zip(table["churned"], table["customers"])]
    table["ci_low"] = [lo for lo, _ in intervals]
    table["ci_high"] = [hi for _, hi in intervals]
    table["share_of_customers"] = table["customers"] / len(df)

    if order is not None and len(columns) == 1:
        present = [g for g in order if g in set(table[columns[0]])]
        table = table.set_index(columns[0]).loc[present].reset_index()
    else:
        table = table.sort_values("churn_rate", ascending=False, ignore_index=True)
    return table


def target_summary(df: pd.DataFrame) -> dict:
    counts = df[TARGET_COLUMN].value_counts()
    return {
        "customers": int(len(df)),
        "churned": int(counts.get("Yes", 0)),
        "retained": int(counts.get("No", 0)),
        "churn_rate": float(df["churned"].mean()),
        "retention_rate": float(1 - df["churned"].mean()),
    }


def data_quality_summary(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Types, blanks and duplicate counts. Uses the raw frame for blanks."""
    return {
        "raw_dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "analysis_dtypes": {c: str(df[c].dtype) for c in NUMERIC_COLUMNS},
        "blank_cells_per_column": {
            c: int((raw[c].str.strip() == "").sum()) for c in raw.columns
            if (raw[c].str.strip() == "").any()
        },
        "total_charges_blank_rows": int(df["TotalCharges_blank"].sum()),
        "total_charges_blank_all_tenure_zero": bool(
            (df.loc[df["TotalCharges_blank"], "tenure"] == 0).all()
        ),
        "tenure_zero_rows": int((df["tenure"] == 0).sum()),
        # How closely TotalCharges matches tenure x MonthlyCharges (tenure > 0 only).
        "total_charges_ratio_to_tenure_x_monthly": {
            k: round(float(v), 3) for k, v in (
                df.loc[df["tenure"] > 0, "TotalCharges"]
                / (df.loc[df["tenure"] > 0, "tenure"] * df.loc[df["tenure"] > 0, "MonthlyCharges"])
            ).quantile([0.05, 0.5, 0.95]).rename({0.05: "p05", 0.5: "median", 0.95: "p95"}).items()
        },
        "duplicate_customer_ids": int(raw[ID_COLUMN].duplicated().sum()),
        "fully_duplicated_rows": int(raw.duplicated().sum()),
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution statistics for each numeric column, overall and by churn."""
    rows = []
    groups = {"all": df, "churned": df[df["churned"] == 1], "retained": df[df["churned"] == 0]}
    for col in NUMERIC_COLUMNS:
        for name, part in groups.items():
            s = part[col]
            rows.append({
                "column": col, "group": name, "count": int(s.count()),
                "missing": int(s.isna().sum()), "mean": s.mean(), "std": s.std(),
                "min": s.min(), "q25": s.quantile(0.25), "median": s.median(),
                "q75": s.quantile(0.75), "max": s.max(), "skew": s.skew(),
            })
    return pd.DataFrame(rows)


def numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations between numeric columns and the churn flag.

    Rows with blank TotalCharges are dropped pairwise by pandas.
    """
    return df[[*NUMERIC_COLUMNS, "churned"]].corr(method="pearson")


def categorical_churn_rates(df: pd.DataFrame) -> pd.DataFrame:
    """One long table: churn rate for every value of every categorical feature."""
    parts = []
    for col in CATEGORICAL_FEATURES:
        t = churn_rate_by(df, col).rename(columns={col: "value"})
        t.insert(0, "feature", col)
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def duplicate_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Find customers who share every feature value (customerID and Churn ignored).

    Nothing is removed. Returns one row per duplicated profile plus a summary.
    A "conflicting" group contains both churners and non-churners, meaning
    the features alone cannot separate those customers.
    """
    dup_mask = df.duplicated(subset=FEATURE_COLUMNS, keep=False)
    dups = df[dup_mask]
    if dups.empty:
        empty = pd.DataFrame(columns=["profile_id", "size", "churned", "retained", "conflicting"])
        return empty, {"groups": 0, "rows": 0, "conflicting_groups": 0, "rows_in_conflicting_groups": 0}

    # dropna=False: TotalCharges is NaN for blank rows; NaN must still group.
    groups = (
        dups.groupby(FEATURE_COLUMNS, dropna=False, observed=True)
        .agg(size=("churned", "size"), churned=("churned", "sum"),
             customer_ids=(ID_COLUMN, lambda s: ";".join(sorted(s))))
        .reset_index()
    )
    groups["retained"] = groups["size"] - groups["churned"]
    groups["conflicting"] = (groups["churned"] > 0) & (groups["retained"] > 0)
    groups = groups.sort_values(["size", "conflicting"], ascending=False, ignore_index=True)
    groups.insert(0, "profile_id", range(1, len(groups) + 1))

    summary = {
        "groups": int(len(groups)),
        "rows": int(groups["size"].sum()),
        "largest_group": int(groups["size"].max()),
        "conflicting_groups": int(groups["conflicting"].sum()),
        "rows_in_conflicting_groups": int(groups.loc[groups["conflicting"], "size"].sum()),
        "share_of_dataset": float(groups["size"].sum() / len(df)),
        "tenure_values_in_duplicated_rows": sorted(int(t) for t in dups["tenure"].unique()),
    }
    columns = ["profile_id", "size", "churned", "retained", "conflicting", *FEATURE_COLUMNS, "customer_ids"]
    return groups[columns], summary
