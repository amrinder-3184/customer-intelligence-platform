"""Tests for churn_platform.features: target, predictors, split, overlap."""

import math

import numpy as np
import pandas as pd
import pytest

from churn_platform.features import MODEL_FEATURES, build_model_frame, profile_overlap, split_data
from test_validation import make_row


def synthetic_raw(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """n valid raw rows with varied values and ~30% churn. Row 0 has a blank TotalCharges."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        tenure = int(rng.integers(1, 72))
        monthly = round(float(rng.uniform(20, 110)), 2)
        rows.append(make_row(
            f"{i:04d}-X",
            tenure=str(tenure),
            MonthlyCharges=str(monthly),
            TotalCharges=str(round(tenure * monthly, 2)),
            Contract=str(rng.choice(["Month-to-month", "One year", "Two year"])),
            gender=str(rng.choice(["Female", "Male"])),
            Churn="Yes" if i % 10 < 3 else "No",
        ))
    rows[0].update(tenure="0", TotalCharges=" ")
    return pd.DataFrame(rows)


def test_target_encoding():
    raw = pd.DataFrame([make_row("A", Churn="Yes"), make_row("B", Churn="No")])
    _, y, _ = build_model_frame(raw)
    assert y.tolist() == [1, 0]


def test_customer_id_and_target_are_not_predictors():
    X, _, ids = build_model_frame(synthetic_raw())
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns and "churned" not in X.columns
    assert list(X.columns) == MODEL_FEATURES
    assert len(MODEL_FEATURES) == 19
    assert ids.is_unique


def test_blank_total_charges_is_nan_and_raw_untouched():
    raw = synthetic_raw()
    before = raw.copy()
    X, _, _ = build_model_frame(raw)
    assert math.isnan(X.loc[0, "TotalCharges"])
    assert X["TotalCharges"].isna().sum() == 1
    pd.testing.assert_frame_equal(raw, before)


def test_split_is_disjoint_and_complete():
    X, y, ids = build_model_frame(synthetic_raw())
    X_tr, X_te, y_tr, y_te, id_tr, id_te = split_data(X, y, ids)
    assert set(id_tr).isdisjoint(id_te)
    assert len(id_tr) + len(id_te) == len(ids)
    assert len(X_te) == 20  # 20% of 100


def test_split_is_stratified():
    X, y, ids = build_model_frame(synthetic_raw(n=200))
    _, _, y_tr, y_te, _, _ = split_data(X, y, ids)
    assert y_tr.mean() == pytest.approx(y.mean(), abs=0.01)
    assert y_te.mean() == pytest.approx(y.mean(), abs=0.01)


def test_split_is_reproducible_with_same_seed():
    X, y, ids = build_model_frame(synthetic_raw())
    first = split_data(X, y, ids, seed=42)[5].tolist()
    second = split_data(X, y, ids, seed=42)[5].tolist()
    other = split_data(X, y, ids, seed=7)[5].tolist()
    assert first == second
    assert first != other


def test_profile_overlap_counts_agreeing_and_conflicting():
    X_train = pd.DataFrame([make_row("A"), make_row("B", tenure="5")])
    X_test = pd.DataFrame([make_row("C"), make_row("D", tenure="5"), make_row("E", tenure="9")])
    Xtr, ytr, _ = build_model_frame(X_train.assign(Churn=["No", "No"]))
    Xte, yte, _ = build_model_frame(X_test.assign(Churn=["No", "Yes", "No"]))
    result = profile_overlap(Xtr, ytr, Xte, yte)
    assert result["test_rows_with_profile_in_train"] == 2   # C matches A, D matches B
    assert result["of_which_all_train_labels_agree"] == 1   # C: No vs No
    assert result["of_which_train_labels_differ_or_mixed"] == 1  # D: Yes vs No
    assert result["overlap_mask"].tolist() == [True, True, False]


def test_profile_overlap_matches_blank_total_charges():
    raw = pd.DataFrame([make_row("A", tenure="0", TotalCharges=" "), make_row("B", tenure="0", TotalCharges=" ")])
    X, y, _ = build_model_frame(raw)
    result = profile_overlap(X.iloc[[0]], y.iloc[[0]], X.iloc[[1]], y.iloc[[1]])
    assert result["test_rows_with_profile_in_train"] == 1
