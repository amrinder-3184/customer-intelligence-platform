"""Tests for interpretation.py and the Phase 4 test-set protections."""

import json

import numpy as np
import pandas as pd
import pytest

from churn_platform import analyze_models
from churn_platform.features import assert_no_test_rows, build_model_frame, split_data
from churn_platform.interpretation import FEATURE_GROUPS, logreg_coefficients, permutation_importance_cv
from churn_platform.model_selection import make_folds
from churn_platform.modeling import build_logreg
from test_validation import make_row


@pytest.fixture(scope="module")
def contract_driven():
    """Churn is fully determined by Contract; everything else is noise."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(300):
        contract = str(rng.choice(["Month-to-month", "One year", "Two year"], p=[0.5, 0.25, 0.25]))
        tenure = int(rng.integers(1, 72))
        monthly = round(float(rng.uniform(20, 110)), 2)
        rows.append(make_row(f"{i:04d}-Z", Contract=contract, tenure=str(tenure), MonthlyCharges=str(monthly),
                             TotalCharges=str(round(tenure * monthly, 2)), gender=str(rng.choice(["Female", "Male"])),
                             Churn="Yes" if contract == "Month-to-month" else "No"))
    X, y, _ = build_model_frame(pd.DataFrame(rows))
    return X, y


def test_coefficient_table_schema_and_reference(contract_driven):
    X, y = contract_driven
    table = logreg_coefficients(build_logreg().fit(X, y), X)
    assert {"feature", "term", "reference", "log_odds", "odds_ratio", "direction", "early_warning"} <= set(table.columns)
    np.testing.assert_allclose(table["odds_ratio"], np.exp(table["log_odds"]))
    assert table["abs_log_odds"].is_monotonic_decreasing
    contract = table[table["feature"] == "Contract"]
    assert set(contract["reference"]) == {"Month-to-month"}                  # most common category
    assert "Month-to-month" not in set(contract["term"])                     # reference has no row of its own
    assert (contract["log_odds"] < 0).all()                                  # other contracts: lower churn odds


def test_numeric_terms_report_training_sd(contract_driven):
    X, y = contract_driven
    table = logreg_coefficients(build_logreg().fit(X, y), X).set_index(["feature", "term"])
    assert table.loc[("tenure", "+1 SD"), "training_sd"] == pytest.approx(X["tenure"].std(ddof=0))
    assert table.loc[("tenure", "+1 SD"), "early_warning"].startswith("retrospective")


def test_permutation_importance_finds_the_driver_and_ignores_noise(contract_driven):
    X, y = contract_driven
    perm = permutation_importance_cv(build_logreg(), X, y, make_folds(), n_repeats=2,
                                     groups=FEATURE_GROUPS).set_index("feature")
    assert perm["auc_drop_mean"].idxmax() == "Contract"
    assert abs(perm.loc["gender", "auc_drop_mean"]) < 0.02
    assert set(FEATURE_GROUPS) <= set(perm.index)


# --- test-set protections --------------------------------------------------------

def test_guard_rejects_test_rows_and_accepts_training_rows(contract_driven):
    X, y = contract_driven
    ids = pd.Series([f"{i}" for i in range(len(X))], index=X.index)
    X_tr, X_te, *_ = split_data(X, y, ids)
    assert_no_test_rows(X_tr, X_te.index)                        # fine
    with pytest.raises(ValueError, match="test rows"):
        assert_no_test_rows(pd.concat([X_tr, X_te.iloc[:1]]), X_te.index)


def test_decisions_hash_ignores_key_order():
    a = {"threshold": 0.16, "model": "logreg"}
    b = {"model": "logreg", "threshold": 0.16}
    assert analyze_models.decisions_hash(a) == analyze_models.decisions_hash(b)
    assert analyze_models.decisions_hash(a) != analyze_models.decisions_hash({**a, "threshold": 0.2})


def test_test_lock(tmp_path, monkeypatch):
    lock = tmp_path / "final_test_evaluation.json"
    monkeypatch.setattr(analyze_models, "FINAL_TEST_FILE", lock)
    assert analyze_models.check_test_lock("abc") is None                 # never scored -> allowed
    lock.write_text(json.dumps({"decisions_sha256": "abc"}), encoding="utf-8")
    assert analyze_models.check_test_lock("abc") is None                 # same decisions -> reproducible re-run
    reason = analyze_models.check_test_lock("different")                 # decisions changed -> refused
    assert reason is not None and "already scored" in reason
