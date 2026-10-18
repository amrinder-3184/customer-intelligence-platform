"""Tests for churn_platform.validation.

The unit tests build a tiny, hand-written DataFrame in memory, so they run
without the real dataset. Each test breaks one thing on purpose and checks
that the validator reports it. This proves the validator can actually fail;
a validator that always says "PASSED" is useless.
"""

import pandas as pd
import pytest

from churn_platform.config import RAW_DATA_FILE
from churn_platform.validation import EXPECTED_COLUMNS, load_raw, summarize, validate


def make_row(customer_id: str, **overrides) -> dict:
    """One valid customer row (all values are strings, like the raw CSV)."""
    row = {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": "12",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": "50.0",
        "TotalCharges": "600.0",
        "Churn": "No",
    }
    row.update(overrides)
    return row


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame([make_row("0001-A"), make_row("0002-B", tenure="30", Churn="Yes")])


def checks(issues, severity=None) -> set:
    return {i.check for i in issues if severity is None or i.severity == severity}


def test_valid_data_has_no_issues(valid_df):
    assert validate(valid_df) == []


def test_missing_column_is_error(valid_df):
    issues = validate(valid_df.drop(columns=["Contract"]))
    assert checks(issues, "ERROR") == {"missing_columns"}


def test_extra_column_is_warning(valid_df):
    issues = validate(valid_df.assign(Notes="x"))
    assert checks(issues, "WARNING") == {"extra_columns"}
    assert checks(issues, "ERROR") == set()


def test_empty_dataframe_is_error():
    issues = validate(pd.DataFrame(columns=EXPECTED_COLUMNS))
    assert "empty" in checks(issues, "ERROR")


def test_duplicate_customer_id_is_error():
    df = pd.DataFrame([make_row("0001-A"), make_row("0001-A", tenure="5")])
    issue = next(i for i in validate(df) if i.check == "duplicate_id")
    assert issue.severity == "ERROR"
    assert issue.row_count == 2


def test_unknown_category_is_error(valid_df):
    valid_df.loc[0, "Contract"] = "Three year"
    assert "invalid_category" in checks(validate(valid_df), "ERROR")


def test_target_must_be_yes_or_no(valid_df):
    valid_df.loc[0, "Churn"] = "maybe"
    assert "invalid_category" in checks(validate(valid_df), "ERROR")


def test_non_numeric_value_is_error(valid_df):
    valid_df.loc[0, "MonthlyCharges"] = "fifty"
    assert "not_numeric" in checks(validate(valid_df), "ERROR")


def test_negative_tenure_is_error(valid_df):
    valid_df.loc[0, "tenure"] = "-1"
    assert "out_of_range" in checks(validate(valid_df), "ERROR")


def test_blank_total_charges_with_zero_tenure_is_warning(valid_df):
    valid_df.loc[0, ["tenure", "TotalCharges"]] = ["0", " "]
    issues = validate(valid_df)
    assert checks(issues, "WARNING") == {"blank_numeric"}
    assert checks(issues, "ERROR") == set()


def test_blank_total_charges_with_positive_tenure_is_error(valid_df):
    valid_df.loc[0, "TotalCharges"] = " "
    assert "blank_numeric" in checks(validate(valid_df), "ERROR")


def test_inconsistent_phone_columns_is_error(valid_df):
    valid_df.loc[0, "PhoneService"] = "No"  # but MultipleLines is still "No"
    assert "inconsistent_phone" in checks(validate(valid_df), "ERROR")


def test_inconsistent_internet_columns_is_error(valid_df):
    valid_df.loc[0, "InternetService"] = "No"  # but add-ons are still Yes/No
    assert "inconsistent_internet" in checks(validate(valid_df), "ERROR")


def test_identical_features_different_ids_is_warning():
    df = pd.DataFrame([make_row("0001-A"), make_row("0002-B", Churn="Yes")])
    assert checks(validate(df)) == {"duplicate_feature_rows"}


def test_summary_churn_rate(valid_df):
    summary = summarize(valid_df)
    assert summary["rows"] == 2
    assert summary["churn_rate"] == 0.5


def test_load_raw_missing_file_gives_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="churn_platform.download"):
        load_raw(tmp_path / "nope.csv")


# Integration test against the real file. Skipped (not failed) when the
# dataset has not been downloaded, e.g. on a fresh clone.
@pytest.mark.skipif(not RAW_DATA_FILE.exists(), reason="raw dataset not downloaded")
def test_real_dataset_has_no_errors():
    df = load_raw(RAW_DATA_FILE)
    assert df.shape == (7043, 21)
    assert checks(validate(df), "ERROR") == set()
