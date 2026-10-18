"""Preprocessing pipelines, models, cross-validation and metrics.

Leakage rule: every step that LEARNS something from data (imputer medians,
scaler means/stds, one-hot category lists, model coefficients) lives inside
a scikit-learn Pipeline. Calling pipeline.fit(X_train, y_train) fits all of
them on training rows only. In cross-validation the whole pipeline is
re-fitted inside each fold, so validation folds never influence
preprocessing either.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_platform.analysis import CATEGORICAL_FEATURES
from churn_platform.config import RANDOM_SEED
from churn_platform.features import NUMERIC_FEATURES

THRESHOLD = 0.5  # fixed default; choosing a business threshold is Phase 4
CV_FOLDS = 5

# Two ways to handle the blank TotalCharges values (all of them tenure-0 customers):
#   "median_indicator": fill with the TRAINING median and add a 0/1 "was missing"
#                       column, so the model can treat those rows differently.
#                       Assumes nothing about what the blank means. (PRIMARY)
#   "zero":             fill with 0, on the assumption that a customer who has
#                       not been billed yet has been charged $0. Plausible
#                       here, but it is a business assumption the data does
#                       not confirm.
IMPUTATION_STRATEGIES = ("median_indicator", "zero")


def build_preprocessor(imputation: str = "median_indicator",
                       numeric_features: list[str] = NUMERIC_FEATURES) -> ColumnTransformer:
    """numeric_features can be narrowed for sensitivity checks (e.g. without tenure)."""
    if imputation == "median_indicator":
        imputer = SimpleImputer(strategy="median", add_indicator=True)
    elif imputation == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    else:
        raise ValueError(f"Unknown imputation strategy: {imputation!r}")

    numeric = Pipeline([
        ("impute", imputer),
        # Logistic regression's L2 penalty treats every coefficient equally, so
        # features need comparable scales. (The 0/1 indicator is scaled too,
        # which is harmless.)
        ("scale", StandardScaler()),
    ])
    # handle_unknown="ignore": a category never seen in training becomes all
    # zeros instead of crashing. Every category is kept (no drop); L2
    # regularisation copes with the redundant columns.
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        [("num", numeric, list(numeric_features)), ("cat", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",  # any column not listed (e.g. a stray customerID) is dropped
        verbose_feature_names_out=True,
    )


def build_logreg(imputation: str = "median_indicator", class_weight=None,
                 numeric_features: list[str] = NUMERIC_FEATURES) -> Pipeline:
    # lbfgs is deterministic; random_state is set anyway for consistency.
    # max_iter raised from 100 so the solver converges on one-hot data.
    # C=1.0 is the library default; it is NOT tuned in Phase 3.
    model = LogisticRegression(C=1.0, max_iter=1000, class_weight=class_weight, random_state=RANDOM_SEED)
    return Pipeline([("preprocess", build_preprocessor(imputation, numeric_features)), ("model", model)])


def build_models(imputation: str = "median_indicator") -> dict[str, Pipeline]:
    """The Phase 3 model set. All share the same preprocessing."""
    return {
        "dummy_most_frequent": Pipeline([
            ("preprocess", build_preprocessor(imputation)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "logreg": build_logreg(imputation),
        # class_weight="balanced" weights each churner ~1.9x and each retained
        # customer ~0.68x, so mistakes on the minority class cost more in training.
        "logreg_balanced": build_logreg(imputation, class_weight="balanced"),
    }


CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def cross_validate_models(models: dict[str, Pipeline], X_train, y_train, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """5-fold stratified CV on the TRAINING set only. Same folds for every model."""
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        # error_score="raise": fail loudly rather than record NaN silently.
        # Dummy has no positive predictions, so precision is 0/0; sklearn
        # warns and scores it 0. That is reported, not hidden.
        scores = cross_validate(model, X_train, y_train, cv=folds, scoring=CV_SCORING,
                                n_jobs=1, error_score="raise")
        row = {"model": name}
        for metric in CV_SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(model: Pipeline, X, y, threshold: float = THRESHOLD) -> dict:
    """All metrics from the model's ACTUAL predictions on (X, y)."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    y = np.asarray(y)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    single_class = len(np.unique(y)) < 2
    return {
        "threshold": threshold,
        "n": int(len(y)),
        "positives": int(y.sum()),
        "predicted_positives": int(pred.sum()),
        "accuracy": float(accuracy_score(y, pred)),
        # zero_division=0: if the model never predicts churn, precision is 0/0.
        # We report 0 and flag it with precision_defined=False.
        "precision": float(precision_score(y, pred, zero_division=0)),
        "precision_defined": bool(pred.sum() > 0),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": None if single_class else float(roc_auc_score(y, proba)),
        "average_precision": None if single_class else float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y, pred, labels=[0, 1], target_names=["Retained", "Churned"], zero_division=0, output_dict=True),
        "classification_report_text": classification_report(
            y, pred, labels=[0, 1], target_names=["Retained", "Churned"], zero_division=0, digits=3),
    }
