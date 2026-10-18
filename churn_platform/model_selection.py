"""Phase 4: compare a few simple models with out-of-fold (OOF) predictions.

Everything here runs on TRAINING rows only. The same 5 stratified folds as
Phase 3 (seed 42) are used for every model, so fold k is the same rows for
all of them, and per-fold differences between models are fair ("paired").

Out-of-fold prediction: for each fold, the whole pipeline (preprocessing +
model) is cloned and fitted on the other 4 folds, then predicts the held-out
fold. Every training row therefore gets exactly one prediction from a model
that never saw its label.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from churn_platform.config import RANDOM_SEED
from churn_platform.modeling import CV_FOLDS, THRESHOLD, build_logreg, build_preprocessor

# Candidate settings are FIXED in advance (no hyperparameter search):
# - Decision tree: depth 5 and >= 20 customers per leaf keep it small enough
#   to read and stop it memorising individual customers.
# - Random forest: 300 trees, depth <= 8, >= 10 customers per leaf. These are
#   conservative, commonly used settings. n_jobs=1 keeps memory low and
#   results deterministic.
# Trees don't need scaling, but sharing the same preprocessor keeps the
# inputs identical across models (scaling doesn't change tree splits).
CANDIDATE_NAMES = ["dummy_most_frequent", "logreg", "decision_tree", "random_forest"]
BASELINE_MODEL = "logreg"
SIMPLICITY_ORDER = ["logreg", "decision_tree", "random_forest"]  # most to least interpretable


def build_candidates() -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline([("preprocess", build_preprocessor()),
                                         ("model", DummyClassifier(strategy="most_frequent"))]),
        "logreg": build_logreg(),  # identical to the Phase 3 primary model
        "decision_tree": Pipeline([("preprocess", build_preprocessor()), ("model", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=RANDOM_SEED))]),
        "random_forest": Pipeline([("preprocess", build_preprocessor()), ("model", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=10, max_features="sqrt",
            n_jobs=1, random_state=RANDOM_SEED))]),
    }


def make_folds(seed: int = RANDOM_SEED) -> StratifiedKFold:
    """Exactly the Phase 3 CV folds."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)


def fold_metrics(y_true, proba, threshold: float = THRESHOLD) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
    }


def oof_evaluate(model, X, y, folds=None) -> tuple[np.ndarray, pd.DataFrame]:
    """OOF churn probabilities plus one row of metrics per fold.

    Returns (oof_proba aligned with X's rows, fold table).
    """
    folds = folds or make_folds()
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        fitted = clone(model)                       # fresh, unfitted copy every fold
        start = time.perf_counter()
        fitted.fit(X.iloc[train_idx], y[train_idx])
        fit_seconds = time.perf_counter() - start
        proba = fitted.predict_proba(X.iloc[val_idx])[:, 1]
        oof[val_idx] = proba
        rows.append({"fold": k, "n_val": len(val_idx), "fit_seconds": fit_seconds,
                     **fold_metrics(y[val_idx], proba)})
    assert not np.isnan(oof).any(), "every training row must get exactly one OOF prediction"
    return oof, pd.DataFrame(rows)


def compare_models(models: dict, X, y, folds=None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fold-level table for every model (same folds) and their OOF probabilities."""
    folds = folds or make_folds()
    tables, oof = [], {}
    for name, model in models.items():
        oof[name], table = oof_evaluate(model, X, y, folds)
        tables.append(table.assign(model=name))
    folds_table = pd.concat(tables, ignore_index=True)
    return folds_table[["model", *[c for c in folds_table.columns if c != "model"]]], oof


def summarize_folds(folds_table: pd.DataFrame) -> pd.DataFrame:
    metrics = ["roc_auc", "average_precision", "precision", "recall", "f1", "accuracy", "fit_seconds"]
    g = folds_table.groupby("model", sort=False)[metrics]
    summary = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std"))
    return summary.reset_index()


def paired_differences(folds_table: pd.DataFrame, baseline: str = BASELINE_MODEL) -> pd.DataFrame:
    """Per-fold (candidate - baseline) for ROC-AUC and AP, with a standard error.

    Caveat: CV folds share training data, so they are not independent and this
    standard error is optimistic. That is why the decision rule asks for a
    margin of 2 SE in BOTH metrics instead of a formal significance test.
    """
    rows = []
    base = folds_table[folds_table["model"] == baseline].set_index("fold")
    for name in folds_table["model"].unique():
        if name == baseline:
            continue
        other = folds_table[folds_table["model"] == name].set_index("fold")
        row = {"model": name}
        for metric in ("roc_auc", "average_precision"):
            diff = other[metric] - base[metric]
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            row.update({f"{metric}_diff_mean": diff.mean(), f"{metric}_diff_se": se,
                        f"{metric}_folds_better": int((diff > 0).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def select_model(diffs: pd.DataFrame, baseline: str = BASELINE_MODEL) -> tuple[str, str]:
    """Pre-declared rule: keep the baseline unless a candidate improves BOTH
    ROC-AUC and AP by more than 2 standard errors of the paired difference.
    If several qualify, take the simplest (SIMPLICITY_ORDER)."""
    qualifying = [
        r["model"] for _, r in diffs.iterrows()
        if r["model"] in SIMPLICITY_ORDER
        and r["roc_auc_diff_mean"] > 2 * r["roc_auc_diff_se"]
        and r["average_precision_diff_mean"] > 2 * r["average_precision_diff_se"]
    ]
    if not qualifying:
        return baseline, "no candidate improved both ROC-AUC and AP by more than 2 SE of the paired fold difference"
    choice = min(qualifying, key=SIMPLICITY_ORDER.index)
    return choice, f"{choice} improved both ROC-AUC and AP by more than 2 SE over {baseline}"
