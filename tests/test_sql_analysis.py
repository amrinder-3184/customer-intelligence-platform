"""Tests for the SQLite layer: view transformations, queries, and SQL/pandas agreement.

Unit tests use an in-memory database (":memory:") so they never touch the
project's database file.
"""

import sqlite3

import pandas as pd
import pytest

from churn_platform.analysis import to_analysis_frame
from churn_platform.config import RAW_DATA_FILE
from churn_platform.sql_analysis import build_database, compare_with_python, load_queries, run_queries
from churn_platform.validation import load_raw
from test_validation import make_row

EXPECTED_QUERIES = {
    "overall_churn", "churn_by_contract", "churn_by_tenure_group", "churn_by_payment_method",
    "monthly_charges_by_churn_status", "churn_by_monthly_charges_band", "segment_contract_internet",
}


@pytest.fixture
def small_raw() -> pd.DataFrame:
    return pd.DataFrame([
        make_row("A", tenure="0", TotalCharges=" ", Churn="No"),
        make_row("B", tenure="12", Contract="Month-to-month", Churn="Yes"),
        make_row("C", tenure="13", Contract="One year", MonthlyCharges="95.5", Churn="No"),
        make_row("D", tenure="49", Contract="Two year", PaymentMethod="Mailed check", Churn="Yes",
                 InternetService="Fiber optic", MonthlyCharges="70"),
    ])


@pytest.fixture
def conn(small_raw):
    connection = sqlite3.connect(":memory:")
    build_database(small_raw, connection)
    yield connection
    connection.close()


def test_query_file_contains_all_expected_queries():
    assert set(load_queries()) == EXPECTED_QUERIES


def test_queries_are_read_only():
    for name, sql in load_queries().items():
        code = [line for line in sql.splitlines() if line.strip() and not line.strip().startswith("--")]
        first_word = code[0].split(None, 1)[0].upper()
        assert first_word == "SELECT", name


def test_raw_table_keeps_blank_text(conn):
    value = conn.execute("SELECT TotalCharges FROM customers_raw WHERE customerID = 'A'").fetchone()[0]
    assert value == " "  # raw value preserved exactly


def test_view_turns_blank_total_charges_into_null(conn):
    value = conn.execute("SELECT total_charges FROM customers WHERE customer_id = 'A'").fetchone()[0]
    assert value is None


def test_view_tenure_groups_match_python(conn, small_raw):
    sql = dict(conn.execute("SELECT customer_id, tenure_group FROM customers").fetchall())
    py = to_analysis_frame(small_raw).set_index("customerID")["tenure_group"].to_dict()
    assert sql == py == {"A": "0-12", "B": "0-12", "C": "13-24", "D": "49+"}


def test_view_charge_bands_match_python(conn, small_raw):
    sql = dict(conn.execute("SELECT customer_id, monthly_charges_band FROM customers").fetchall())
    py = to_analysis_frame(small_raw).set_index("customerID")["monthly_charges_band"].to_dict()
    assert sql == py


def test_overall_churn_query(conn):
    row = run_queries(conn)["overall_churn"].iloc[0]
    assert (row["customers"], row["churned"], row["retained"]) == (4, 2, 2)
    assert row["churn_rate"] == 0.5


def test_sql_matches_python_on_small_data(conn, small_raw):
    checks = compare_with_python(run_queries(conn), to_analysis_frame(small_raw))
    assert {c["query"] for c in checks} == EXPECTED_QUERIES
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]


def test_comparison_detects_a_mismatch(conn, small_raw):
    results = run_queries(conn)
    results["churn_by_contract"].loc[0, "churned"] += 1   # tamper with one SQL number
    failed = [c["query"] for c in compare_with_python(results, to_analysis_frame(small_raw)) if not c["passed"]]
    assert failed == ["churn_by_contract"]


def test_build_database_is_repeatable(small_raw):
    connection = sqlite3.connect(":memory:")
    build_database(small_raw, connection)
    build_database(small_raw, connection)  # second build must replace, not append
    assert connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 4
    connection.close()


@pytest.mark.skipif(not RAW_DATA_FILE.exists(), reason="raw dataset not downloaded")
def test_sql_matches_python_on_real_data():
    raw = load_raw(RAW_DATA_FILE)
    connection = sqlite3.connect(":memory:")
    build_database(raw, connection)
    results = run_queries(connection)
    connection.close()
    assert results["overall_churn"].iloc[0]["customers"] == 7043
    checks = compare_with_python(results, to_analysis_frame(raw))
    assert all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
