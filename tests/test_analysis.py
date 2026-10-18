"""Tests for churn_platform.analysis (transformations and churn calculations)."""

import math

import pandas as pd
import pytest

from churn_platform.analysis import (
    FEATURE_COLUMNS,
    churn_rate_by,
    duplicate_profiles,
    target_summary,
    to_analysis_frame,
    wilson_interval,
)
from test_validation import make_row  # same valid-row builder as the Phase 1 tests


def frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# --- to_analysis_frame ------------------------------------------------------

def test_blank_total_charges_becomes_nan_not_zero():
    raw = frame(make_row("A", tenure="0", TotalCharges=" "), make_row("B"))
    df = to_analysis_frame(raw)
    assert math.isnan(df.loc[0, "TotalCharges"])
    assert df.loc[1, "TotalCharges"] == 600.0
    assert df["TotalCharges_blank"].tolist() == [True, False]


def test_raw_frame_is_not_modified():
    raw = frame(make_row("A", tenure="0", TotalCharges=" "))
    before = raw.copy()
    to_analysis_frame(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_row_count_preserved_and_types():
    raw = frame(make_row("A"), make_row("B", tenure="0", TotalCharges=" "), make_row("C", Churn="Yes"))
    df = to_analysis_frame(raw)
    assert len(df) == len(raw)
    assert df["tenure"].dtype == "int64"
    assert df["MonthlyCharges"].dtype == "float64"
    assert df["TotalCharges"].dtype == "float64"
    assert df["churned"].tolist() == [0, 0, 1]


def test_non_numeric_total_charges_raises_instead_of_silently_becoming_nan():
    with pytest.raises(ValueError):
        to_analysis_frame(frame(make_row("A", TotalCharges="abc")))


@pytest.mark.parametrize("tenure, group", [
    (0, "0-12"), (12, "0-12"), (13, "13-24"), (24, "13-24"),
    (25, "25-48"), (48, "25-48"), (49, "49+"), (72, "49+"),
])
def test_tenure_group_boundaries(tenure, group):
    df = to_analysis_frame(frame(make_row("A", tenure=str(tenure))))
    assert df.loc[0, "tenure_group"] == group


@pytest.mark.parametrize("charge, band", [
    ("18.25", "<35"), ("34.99", "<35"), ("35", "35-70"), ("69.99", "35-70"),
    ("70", "70-90"), ("89.99", "70-90"), ("90", "90+"), ("118.75", "90+"),
])
def test_monthly_charge_band_boundaries(charge, band):
    df = to_analysis_frame(frame(make_row("A", MonthlyCharges=charge)))
    assert df.loc[0, "monthly_charges_band"] == band


def test_customer_id_is_not_a_feature():
    assert "customerID" not in FEATURE_COLUMNS
    assert "Churn" not in FEATURE_COLUMNS


# --- churn rates ------------------------------------------------------------

def test_churn_rate_by_counts_and_rates():
    raw = frame(
        make_row("A", Contract="Month-to-month", Churn="Yes"),
        make_row("B", Contract="Month-to-month", Churn="No"),
        make_row("C", Contract="Two year", Churn="No"),
    )
    t = churn_rate_by(to_analysis_frame(raw), "Contract").set_index("Contract")
    assert t.loc["Month-to-month", "customers"] == 2
    assert t.loc["Month-to-month", "churned"] == 1
    assert t.loc["Month-to-month", "churn_rate"] == 0.5
    assert t.loc["Two year", "churn_rate"] == 0.0
    assert t["customers"].sum() == 3


def test_churn_rate_by_respects_order():
    raw = frame(make_row("A", tenure="60"), make_row("B", tenure="1", Churn="Yes"))
    t = churn_rate_by(to_analysis_frame(raw), "tenure_group", ["49+", "0-12"])
    assert t["tenure_group"].tolist() == ["49+", "0-12"]


def test_churn_rate_by_order_skips_empty_groups():
    # Regression: an order listing a group with no customers used to raise KeyError.
    raw = frame(make_row("A", tenure="60"), make_row("B", tenure="1"))
    t = churn_rate_by(to_analysis_frame(raw), "tenure_group", ["0-12", "13-24", "25-48", "49+"])
    assert t["tenure_group"].tolist() == ["0-12", "49+"]


def test_target_summary():
    raw = frame(make_row("A", Churn="Yes"), make_row("B"), make_row("C"), make_row("D"))
    s = target_summary(to_analysis_frame(raw))
    assert (s["customers"], s["churned"], s["retained"]) == (4, 1, 3)
    assert s["churn_rate"] == 0.25


def test_wilson_interval_known_value():
    # 50 of 100: standard result is approximately (0.404, 0.596).
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_interval_stays_within_0_and_1():
    assert wilson_interval(0, 10)[0] >= 0
    assert wilson_interval(10, 10)[1] <= 1


# --- duplicate profiles -----------------------------------------------------

def test_duplicate_profiles_detects_conflicting_group_and_keeps_rows():
    raw = frame(
        make_row("A", Churn="Yes"),
        make_row("B", Churn="No"),          # same features as A, other outcome
        make_row("C", tenure="40"),          # unique
        make_row("D", tenure="50"), make_row("E", tenure="50"),  # duplicate, same outcome
    )
    df = to_analysis_frame(raw)
    table, summary = duplicate_profiles(df)
    assert len(df) == 5                       # nothing removed
    assert summary["groups"] == 2
    assert summary["rows"] == 4
    assert summary["conflicting_groups"] == 1
    assert summary["rows_in_conflicting_groups"] == 2
    conflicting = table[table["conflicting"]].iloc[0]
    assert conflicting["customer_ids"] == "A;B"


def test_duplicate_profiles_groups_blank_total_charges_together():
    # NaN TotalCharges must still be grouped (pandas drops NaN keys by default).
    raw = frame(make_row("A", tenure="0", TotalCharges=" "), make_row("B", tenure="0", TotalCharges=" "))
    _, summary = duplicate_profiles(to_analysis_frame(raw))
    assert summary["groups"] == 1


def test_no_duplicates():
    raw = frame(make_row("A"), make_row("B", tenure="5"))
    table, summary = duplicate_profiles(to_analysis_frame(raw))
    assert table.empty and summary["groups"] == 0
