"""Phase 4: how trustworthy are the predicted probabilities?

A model is CALIBRATED if, among customers it gives ~30% churn probability,
about 30% actually churn. Ranking (ROC-AUC) and calibration are different:
a model can rank perfectly yet output probabilities that are all too high.

Calibration matters here because the cost-based threshold (thresholds.py)
and any "expected cost" estimate treat the scores as real probabilities.

Isolation: calibrated variants are evaluated with the SAME outer OOF loop as
everything else. CalibratedClassifierCV runs its own inner 5-fold CV inside
each outer training part, so the calibrator never sees the outer validation
fold's labels.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from churn_platform.config import RANDOM_SEED

CALIBRATION_METHODS = ("sigmoid", "isotonic")
N_BINS = 10


def build_calibrated(model, method: str) -> CalibratedClassifierCV:
    """Wrap an (unfitted) pipeline. Inner CV: 5 stratified folds, fixed seed.

    sigmoid  (Platt scaling): fits a 2-parameter S-curve to the scores. Stable
             with little data, but can only fix S-shaped distortions.
    isotonic: fits any increasing step function. More flexible, but it can
             overfit when the data is small.
    """
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    return CalibratedClassifierCV(estimator=model, method=method, cv=inner)


def expected_calibration_error(y_true, proba, n_bins: int = N_BINS) -> float:
    """Weighted average gap |mean predicted - observed rate| over quantile bins.

    Quantile bins hold equal numbers of customers, so no bin is nearly empty.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, proba, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        ece += in_bin.mean() * abs(proba[in_bin].mean() - y_true[in_bin].mean())
    return float(ece)


def probability_metrics(y_true, proba) -> dict:
    """Brier, log loss, ECE, and Brier relative to a no-skill reference.

    The no-skill reference predicts the overall churn rate p for everyone, giving
    Brier = p(1-p). Brier depends on the event rate, so a raw Brier value is
    only meaningful next to that reference. Skill = 1 - Brier/reference.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    p = y_true.mean()
    brier = brier_score_loss(y_true, proba)
    reference = p * (1 - p)
    return {
        "brier": float(brier),
        "brier_reference_no_skill": float(reference),
        # Undefined when the subset contains a single class (reference = 0).
        "brier_skill_score": float(1 - brier / reference) if reference > 0 else None,
        # labels=[0, 1] keeps log loss defined even for a single-class subset.
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "ece": expected_calibration_error(y_true, proba),
        "mean_predicted": float(proba.mean()),
        "observed_rate": float(p),
    }


def per_fold_probability_metrics(y_true, proba, folds, X) -> pd.DataFrame:
    """Brier and log loss separately on each outer validation fold."""
    y_true = np.asarray(y_true)
    rows = []
    for k, (_, val_idx) in enumerate(folds.split(X, y_true)):
        rows.append({"fold": k, "brier": brier_score_loss(y_true[val_idx], proba[val_idx]),
                     "log_loss": log_loss(y_true[val_idx], np.clip(proba[val_idx], 1e-15, 1 - 1e-15), labels=[0, 1])})
    return pd.DataFrame(rows)


def curve_table(y_true, proba, name: str, n_bins: int = N_BINS) -> pd.DataFrame:
    observed, predicted = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"variant": name, "bin": range(len(observed)),
                         "mean_predicted": predicted, "observed_rate": observed})


def select_calibration(per_fold: dict[str, pd.DataFrame], overall: dict[str, dict]) -> tuple[str, str]:
    """Pre-declared rule: adopt a calibrator only if, versus "uncalibrated", it
    lowers OOF Brier AND log loss, and lowers Brier in at least 4 of 5 folds.
    If both qualify, take the lower Brier."""
    base = overall["uncalibrated"]
    qualifying = []
    for method in CALIBRATION_METHODS:
        if method not in overall:
            continue
        better_folds = int((per_fold[method]["brier"].to_numpy() < per_fold["uncalibrated"]["brier"].to_numpy()).sum())
        if (overall[method]["brier"] < base["brier"] and overall[method]["log_loss"] < base["log_loss"]
                and better_folds >= 4):
            qualifying.append(method)
    if not qualifying:
        return "uncalibrated", ("no calibrator lowered OOF Brier and log loss while also lowering Brier "
                                "in at least 4 of 5 folds")
    best = min(qualifying, key=lambda m: overall[m]["brier"])
    return best, f"{best} lowered OOF Brier and log loss and Brier in at least 4 of 5 folds"
