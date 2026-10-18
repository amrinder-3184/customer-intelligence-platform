"""Tests for model_selection.py and calibration.py: OOF isolation, schemas, decision rules."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from churn_platform.calibration import (
    build_calibrated,
    expected_calibration_error,
    probability_metrics,
    select_calibration,
)
from churn_platform.model_selection import (
    compare_models,
    make_folds,
    oof_evaluate,
    paired_differences,
    select_model,
    summarize_folds,
)


class Recorder(ClassifierMixin, BaseEstimator):  # sklearn requires mixins before BaseEstimator
    """Remembers which row_ids every fit() saw (class-level log), and scores
    rows it was trained on as 1.0 and unseen rows by a fixed formula < 1."""
    fits: list = []

    def fit(self, X, y):
        self.seen_ = set(X["row_id"])
        Recorder.fits.append(self.seen_)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.array([1.0 if r in self.seen_ else (r % 7) / 10 for r in X["row_id"]])
        return np.column_stack([1 - p, p])


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"row_id": np.arange(n), "x": rng.normal(size=n)})
    y = pd.Series((X["x"] + rng.normal(scale=0.8, size=n) > 0.6).astype(int))
    return X, y


def test_folds_are_the_phase3_folds(toy):
    X, y = toy
    ours = [v.tolist() for _, v in make_folds().split(X, y)]
    phase3 = [v.tolist() for _, v in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)]
    assert ours == phase3


def test_oof_predicts_every_row_once_with_a_model_that_never_saw_it(toy):
    X, y = toy
    Recorder.fits = []
    oof, table = oof_evaluate(Recorder(), X, y)
    assert len(oof) == len(X) and not np.isnan(oof).any()
    assert (oof < 1.0).all()          # 1.0 would mean the row was in that model's training data
    assert len(table) == 5 and table["n_val"].sum() == len(X)


def test_calibrator_is_nested_inside_outer_training_folds(toy):
    """25 fits = 5 outer folds x 5 inner calibration folds; none may see outer validation rows."""
    X, y = toy
    Recorder.fits = []
    oof_evaluate(build_calibrated(Recorder(), "sigmoid"), X, y)
    assert len(Recorder.fits) == 25
    for k, (train_idx, val_idx) in enumerate(make_folds().split(X, y)):
        val_ids = set(X["row_id"].iloc[val_idx])
        for seen in Recorder.fits[5 * k: 5 * k + 5]:
            assert seen.isdisjoint(val_ids)


def _models():
    return {"logreg": LogisticRegression(), "weak": LogisticRegression(C=1e-6)}


def test_compare_models_schema_and_reproducibility(toy):
    X, y = toy
    X = X[["x"]]
    t1, oof1 = compare_models(_models(), X, y)
    t2, oof2 = compare_models(_models(), X, y)
    expected = {"model", "fold", "n_val", "fit_seconds", "roc_auc", "average_precision",
                "precision", "recall", "f1", "accuracy"}
    assert set(t1.columns) == expected
    assert len(t1) == 2 * 5
    cols = [c for c in t1.columns if c != "fit_seconds"]          # timing is the only non-deterministic column
    pd.testing.assert_frame_equal(t1[cols], t2[cols])
    np.testing.assert_array_equal(oof1["logreg"], oof2["logreg"])
    summary = summarize_folds(t1)
    assert {"roc_auc_mean", "roc_auc_std", "average_precision_mean", "fit_seconds_mean"} <= set(summary.columns)


def _fold_table(base, other):
    rows = [{"model": "logreg", "fold": k, "roc_auc": v, "average_precision": v} for k, v in enumerate(base)]
    rows += [{"model": "random_forest", "fold": k, "roc_auc": v, "average_precision": v} for k, v in enumerate(other)]
    return pd.DataFrame(rows)


def test_paired_differences_values():
    d = paired_differences(_fold_table([0.80, 0.82, 0.84, 0.86, 0.88], [0.81, 0.83, 0.85, 0.87, 0.89]))
    row = d.set_index("model").loc["random_forest"]
    assert row["roc_auc_diff_mean"] == pytest.approx(0.01)
    assert row["roc_auc_diff_se"] == pytest.approx(0.0, abs=1e-12)
    assert row["roc_auc_folds_better"] == 5


def test_select_model_keeps_baseline_when_gain_is_noise():
    noisy = paired_differences(_fold_table([0.80, 0.82, 0.84, 0.86, 0.88], [0.82, 0.81, 0.85, 0.85, 0.89]))
    assert select_model(noisy)[0] == "logreg"


def test_select_model_switches_on_consistent_gain():
    clear = paired_differences(_fold_table([0.80, 0.82, 0.84, 0.86, 0.88], [0.83, 0.85, 0.86, 0.89, 0.90]))
    assert select_model(clear)[0] == "random_forest"


# --- calibration metrics and rule ----------------------------------------------

def test_probability_metrics_perfect_and_no_skill():
    y = np.array([0, 0, 0, 1])
    perfect = probability_metrics(y, y.astype(float))
    assert perfect["brier"] == pytest.approx(0.0)
    no_skill = probability_metrics(y, np.full(4, 0.25))
    assert no_skill["brier"] == pytest.approx(0.25 * 0.75)
    assert no_skill["brier_skill_score"] == pytest.approx(0.0)
    assert no_skill["ece"] == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    y = np.array([0, 1] * 50)                       # true rate 0.5 everywhere
    assert expected_calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.4)


def test_single_class_subset_edge_case():
    m = probability_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert np.isfinite(m["log_loss"])
    assert m["brier_skill_score"] is None          # reference Brier is 0 -> skill undefined, not a crash/inf


def _fold_briers(values):
    return pd.DataFrame({"fold": range(5), "brier": values, "log_loss": values})


def test_select_calibration_requires_fold_consistency():
    overall = {"uncalibrated": {"brier": 0.135, "log_loss": 0.42},
               "sigmoid": {"brier": 0.134, "log_loss": 0.41},
               "isotonic": {"brier": 0.140, "log_loss": 0.43}}
    per_fold = {"uncalibrated": _fold_briers([0.13] * 5),
                "sigmoid": _fold_briers([0.12, 0.12, 0.14, 0.14, 0.14]),   # better in only 2/5 folds
                "isotonic": _fold_briers([0.14] * 5)}
    assert select_calibration(per_fold, overall)[0] == "uncalibrated"
    per_fold["sigmoid"] = _fold_briers([0.12, 0.12, 0.12, 0.12, 0.14])     # better in 4/5
    assert select_calibration(per_fold, overall)[0] == "sigmoid"
