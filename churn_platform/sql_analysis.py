"""Build the SQLite analysis database and run the churn queries.

Usage (from the project root):
    python -m churn_platform.sql_analysis     # rebuild DB, print every query result

Flow:
    raw CSV --(load_raw, all text)--> table customers_raw
            --(sql/create_analysis_view.sql)--> view customers (typed)
            --(sql/churn_queries.sql)--> result tables

The database file is fully derived from the raw CSV and is deleted and
rebuilt on every run, so it can never drift out of date.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

from churn_platform.analysis import churn_rate_by, CHARGE_BANDS, TENURE_GROUPS
from churn_platform.config import ANALYSIS_DB_FILE, RAW_DATA_FILE, SQL_DIR
from churn_platform.validation import load_raw

VIEW_FILE = SQL_DIR / "create_analysis_view.sql"
QUERY_FILE = SQL_DIR / "churn_queries.sql"


def build_database(raw: pd.DataFrame, conn: sqlite3.Connection) -> None:
    """Load the raw text frame into customers_raw and create the typed view."""
    conn.execute("DROP VIEW IF EXISTS customers")
    conn.execute("DROP TABLE IF EXISTS customers_raw")
    # Column names come from our own validated schema, not from a user, and
    # are quoted. Values are passed with ? placeholders, never formatted in.
    cols = ", ".join(f'"{c}" TEXT' for c in raw.columns)
    conn.execute(f"CREATE TABLE customers_raw ({cols})")
    placeholders = ", ".join("?" for _ in raw.columns)
    conn.executemany(
        f"INSERT INTO customers_raw VALUES ({placeholders})",
        raw.itertuples(index=False, name=None),
    )
    conn.executescript(VIEW_FILE.read_text(encoding="utf-8"))
    conn.commit()


def load_queries(path: Path = QUERY_FILE) -> dict[str, str]:
    """Split a .sql file into {name: query} using '-- name: xxx' markers."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.MULTILINE)
    # parts = [header, name1, body1, name2, body2, ...]
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def run_queries(conn: sqlite3.Connection, queries: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    queries = queries if queries is not None else load_queries()
    return {name: pd.read_sql_query(sql, conn) for name, sql in queries.items()}


def _python_equivalents(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The same aggregations as churn_queries.sql, computed with pandas."""
    def rate_table(col, sql_col, order=None):
        t = churn_rate_by(df, col, order)
        return t.rename(columns={col: sql_col})[[sql_col, "customers", "churned", "churn_rate"]]

    status = df.assign(churn_status=df["churned"].map({1: "Churned", 0: "Retained"}))
    monthly = (
        status.groupby("churn_status")["MonthlyCharges"]
        .agg(customers="count", avg_monthly_charges="mean",
             min_monthly_charges="min", max_monthly_charges="max")
        .reset_index()
    )
    seg = churn_rate_by(df, ["Contract", "InternetService"]).rename(
        columns={"Contract": "contract", "InternetService": "internet_service"})
    seg["pct_of_customers"] = 100 * seg["share_of_customers"]

    return {
        "overall_churn": pd.DataFrame([{
            "customers": len(df), "churned": int(df["churned"].sum()),
            "retained": int((df["churned"] == 0).sum()), "churn_rate": df["churned"].mean(),
        }]),
        "churn_by_contract": rate_table("Contract", "contract"),
        "churn_by_tenure_group": rate_table("tenure_group", "tenure_group", TENURE_GROUPS),
        "churn_by_payment_method": rate_table("PaymentMethod", "payment_method"),
        "monthly_charges_by_churn_status": monthly,
        "churn_by_monthly_charges_band": rate_table("monthly_charges_band", "monthly_charges_band", CHARGE_BANDS),
        "segment_contract_internet": seg[["contract", "internet_service", "customers",
                                          "pct_of_customers", "churned", "churn_rate"]],
    }


def compare_with_python(sql_results: dict[str, pd.DataFrame], df: pd.DataFrame) -> list[dict]:
    """Check every SQL result against the pandas calculation.

    Text columns are join keys; integer columns must match exactly; decimal
    columns must match within the rounding used in the SQL (4 dp for rates,
    2 dp for money and percentages).
    """
    checks = []
    for name, py in _python_equivalents(df).items():
        sql = sql_results[name]
        keys = [c for c in sql.columns if sql[c].dtype == object or pd.api.types.is_string_dtype(sql[c])]
        if not keys:  # e.g. overall_churn: a single row, nothing to join on
            sql, py, keys = sql.assign(_row=range(len(sql))), py.assign(_row=range(len(py))), ["_row"]
        merged = sql.merge(py, on=keys, how="outer", suffixes=("_sql", "_py"), indicator=True)
        problems = []
        if (merged["_merge"] != "both").any() or len(sql) != len(py):
            problems.append("different groups in SQL and Python")
        for col in [c for c in sql.columns if c not in keys]:
            a, b = merged[f"{col}_sql"].astype(float), merged[f"{col}_py"].astype(float)
            tol = 0 if pd.api.types.is_integer_dtype(sql[col]) else (6e-5 if "rate" in col else 6e-3)
            worst = float((a - b).abs().max())
            if not worst <= tol:  # "not <=" also catches NaN
                problems.append(f"{col}: max difference {worst}")
        checks.append({"query": name, "rows": int(len(sql)), "passed": not problems, "problems": problems})
    return checks


def build_and_query(raw: pd.DataFrame, db_path: Path = ANALYSIS_DB_FILE) -> dict[str, pd.DataFrame]:
    """Rebuild the database file from scratch and return all query results."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        build_database(raw, conn)
        results = run_queries(conn)
    conn.close()  # the "with" block commits but does not close on its own
    return results


def main() -> None:
    raw = load_raw(RAW_DATA_FILE)
    results = build_and_query(raw)
    print(f"Database: {ANALYSIS_DB_FILE}\n")
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        for name, table in results.items():
            print(f"-- {name}\n{table.to_string(index=False)}\n")


if __name__ == "__main__":
    main()
