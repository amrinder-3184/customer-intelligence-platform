"""Tests for churn_platform.modeling: leakage safety, missing values, metrics, reproducibility."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from churn_platform.config import RAW_DATA_FILE
from churn_platform.features import build_model_frame, split_data
from churn_platform.modeling import (
    CV_SCORING,
    build_logreg,
    build_models,
    build_preprocessor,
    cross_validate_models,
    evaluate,
)
from churn_platform.validation import load_raw
from test_features import synthetic_raw
from test_validation import make_row


@pytest.fixture
def split():
    X, y, ids = build_model_frame(synthetic_raw(n=200))
    return split_data(X, y, ids)


# --- fitted on training data only ------------------------------------------

def test_scaler_and_imputer_learn_from_training_rows_only(split):
    X_tr, X_te, *_ = split
    pre = build_preprocessor().fit(X_tr)
    num = pre.named_transformers_["num"]
    imputer, scaler = num.named_steps["impute"], num.named_steps["scale"]

    assert imputer.statistics_[0] == pytest.approx(X_tr["tenure"].median())
    assert imputer.statistics_[2] == pytest.approx(X_tr["TotalCharges"].median())   # NaN ignored
    assert scaler.mean_[1] == pytest.approx(X_tr["MonthlyCharges"].mean())
    # And NOT the statistics of all data combined:
    all_mean = pd.concat([X_tr, X_te])["MonthlyCharges"].mean()
    assert scaler.mean_[1] != pytest.approx(all_mean, abs=1e-9)


def test_category_seen_only_in_test_is_not_learned_and_does_not_crash():
    X_tr, _, _ = build_model_frame(pd.DataFrame([make_row("A", PaymentMethod="Mailed check"),
                                                 make_row("B", PaymentMethod="Electronic check")]))
    X_te, _, _ = build_model_frame(pd.DataFrame([make_row("C", PaymentMethod="Credit card (automatic)")]))
    pre = build_preprocessor().fit(X_tr)
    encoder = pre.named_transformers_["cat"]
    learned = encoder.categories_[list(encoder.feature_names_in_).index("PaymentMethod")]
    assert "Credit card (automatic)" not in learned
    out = pd.DataFrame(pre.transform(X_te), columns=pre.get_feature_names_out())
    payment_cols = [c for c in out.columns if c.startswith("cat__PaymentMethod_")]
    assert out[payment_cols].to_numpy().sum() == 0   # unseen category -> all zeros


# --- missing TotalCharges ----------------------------------------------------

@pytest.fixture
def frame_with_blank():
    """Full synthetic frame (not split), so the one blank TotalCharges row is guaranteed present."""
    X, y, _ = build_model_frame(synthetic_raw(n=200))
    return X, y


def test_median_indicator_strategy_fills_and_flags(frame_with_blank):
    X_tr, _ = frame_with_blank
    pre = build_preprocessor("median_indicator").fit(X_tr)
    out = pd.DataFrame(pre.transform(X_tr), columns=pre.get_feature_names_out(), index=X_tr.index)
    assert not out.isna().any().any()
    flag = out["num__missingindicator_TotalCharges"]
    blank = X_tr["TotalCharges"].isna()
    assert blank.sum() == 1
    # The indicator is scaled, so check that the blank row stands apart from the rest.
    assert flag[blank].iloc[0] > flag[~blank].max()


def test_indicator_only_exists_if_training_data_had_missing_values(frame_with_blank):
    # Documented sklearn behaviour: add_indicator creates the column only for
    # features that were missing during fit. Output columns are fixed at fit
    # time, so prediction still works (a later blank is just median-filled).
    X, _ = frame_with_blank
    complete = X.dropna()
    pre = build_preprocessor("median_indicator").fit(complete)
    assert not any("missingindicator" in c for c in pre.get_feature_names_out())
    out = pre.transform(X)   # contains a blank -> must not crash or produce NaN
    assert not np.isnan(out).any()


def test_zero_strategy_fills_with_zero_and_adds_no_indicator(frame_with_blank):
    X_tr, _ = frame_with_blank
    pre = build_preprocessor("zero").fit(X_tr)
    imputed = pre.named_transformers_["num"].named_steps["impute"].transform(X_tr[["tenure", "MonthlyCharges", "TotalCharges"]])
    blank = X_tr["TotalCharges"].isna().to_numpy()
    assert imputed[blank, 2].tolist() == [0.0]
    assert not any("missingindicator" in c for c in pre.get_feature_names_out())


def test_unknown_imputation_strategy_rejected():
    with pytest.raises(ValueError):
        build_preprocessor("mean_of_everything")


# --- identifier exclusion ----------------------------------------------------

def test_customer_id_column_is_dropped_even_if_passed(split):
    X_tr, _, y_tr, _, id_tr, _ = split
    leaky = X_tr.assign(customerID=id_tr.values)
    model = build_logreg().fit(leaky, y_tr)
    names = model.named_steps["preprocess"].get_feature_names_out()
    assert not any("customerID" in n for n in names)


# --- metrics -----------------------------------------------------------------

class FixedProbabilities:
    """Stand-in model that returns preset churn probabilities."""
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


def test_evaluate_computes_metrics_from_predictions():
    y = [1, 1, 0, 0, 0]
    m = evaluate(FixedProbabilities([0.9, 0.4, 0.6, 0.2, 0.1]), None, y)
    # threshold 0.5 -> predictions [1, 0, 1, 0, 0]
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 2)
    assert m["f1"] == pytest.approx(1 / 2)
    assert m["roc_auc"] == pytest.approx(5 / 6)   # 5 of 6 (churner, non-churner) pairs ranked correctly
    assert m["precision_defined"] is True
    assert "Churned" in m["classification_report"]


def test_evaluate_flags_undefined_precision_for_never_positive_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = evaluate(FixedProbabilities([0.2, 0.2, 0.2]), None, [1, 0, 0])
    assert m["predicted_positives"] == 0
    assert m["precision_defined"] is False
    assert m["recall"] == 0.0
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_evaluate_auc_undefined_for_single_class():
    m = evaluate(FixedProbabilities([0.7, 0.2]), None, [0, 0])
    assert m["roc_auc"] is None and m["average_precision"] is None


def test_cross_validation_returns_every_metric(split):
    X_tr, _, y_tr, *_ = split
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = cross_validate_models(build_models(), X_tr, y_tr)
    assert set(cv["model"]) == {"dummy_most_frequent", "logreg", "logreg_balanced"}
    for metric in CV_SCORING:
        assert f"{metric}_mean" in cv.columns and f"{metric}_std" in cv.columns
    assert cv.set_index("model").loc["dummy_most_frequent", "roc_auc_mean"] == 0.5


# --- real data: convergence and reproducibility -------------------------------

needs_data = pytest.mark.skipif(not RAW_DATA_FILE.exists(), reason="raw dataset not downloaded")


def _run_once():
    """Fit on the TRAINING partition and score the same training rows.

    The held-out test parts of the split are discarded (`_`): the test set was
    already evaluated in Phases 3-4 and must not be re-scored by unit tests.
    Training-row metrics are fine for a reproducibility check (identical inputs
    must give identical outputs); they are not a measure of generalisation.
    """
    X, y, ids = build_model_frame(load_raw(RAW_DATA_FILE))
    X_tr, _, y_tr, _, id_tr, _ = split_data(X, y, ids)
    model = build_logreg().fit(X_tr, y_tr)
    proba = model.predict_proba(X_tr)[:, 1]
    return (id_tr.tolist(), model.named_steps["model"].coef_.copy(), proba,
            evaluate(model, X_tr, y_tr))


@needs_data
def test_real_data_logreg_converges_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        _run_once()


@needs_data
def test_real_data_pipeline_is_reproducible():
    ids_1, coef_1, proba_1, metrics_1 = _run_once()
    ids_2, coef_2, proba_2, metrics_2 = _run_once()
    assert ids_1 == ids_2                                   # same training partition
    assert metrics_1["n"] == len(ids_1) == 5634             # scored rows = training rows
    np.testing.assert_array_equal(coef_1, coef_2)
    np.testing.assert_array_equal(proba_1, proba_2)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
        assert metrics_1[key] == metrics_2[key]
    assert metrics_1["confusion_matrix"] == metrics_2["confusion_matrix"]


@needs_data
def test_real_data_helper_never_touches_test_rows(monkeypatch):
    """Regression guard: _run_once must fit and predict on training rows only.

    Every Pipeline.fit / Pipeline.predict_proba call is recorded with the row
    index it received. If someone reintroduces evaluate(model, X_te, y_te) or
    any prediction on test rows, a test-row index appears here and this fails.
    """
    from sklearn.pipeline import Pipeline

    X, y, ids = build_model_frame(load_raw(RAW_DATA_FILE))
    X_tr, X_te, *_ = split_data(X, y, ids)
    train_rows, test_rows = set(X_tr.index), set(X_te.index)

    seen = {"fit": [], "predict_proba": []}
    original_fit, original_proba = Pipeline.fit, Pipeline.predict_proba

    def recording_fit(self, X, y=None, **kw):
        seen["fit"].append(set(X.index))
        return original_fit(self, X, y, **kw)

    def recording_proba(self, X, **kw):
        seen["predict_proba"].append(set(X.index))
        return original_proba(self, X, **kw)

    monkeypatch.setattr(Pipeline, "fit", recording_fit)
    monkeypatch.setattr(Pipeline, "predict_proba", recording_proba)
    _run_once()

    assert seen["fit"] and seen["predict_proba"]            # the check is not vacuous
    for rows in seen["fit"] + seen["predict_proba"]:
        assert rows <= train_rows
        assert rows.isdisjoint(test_rows)
