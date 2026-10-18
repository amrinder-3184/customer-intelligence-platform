"""Phase 4: which features are associated with the model's predictions?

Two complementary views:

1. LOGISTIC REGRESSION COEFFICIENTS (fitted on the training set).
   The model predicts log-odds = intercept + sum(coef * input). exp(coef) is
   an ODDS RATIO: the factor by which the odds of churn are multiplied.
   - Numeric inputs are standardised, so a coefficient means "per 1 standard
     deviation" (SD measured on the training set). We also show the SD.
   - Categorical inputs are one-hot encoded with EVERY category kept. With an
     intercept, only DIFFERENCES between categories of the same feature are
     meaningful (adding a constant to all of them changes nothing). So each
     category is reported relative to a reference category (the most common
     one in training).
   - Correlated inputs share credit unpredictably (e.g. tenure and
     TotalCharges), so individual coefficients can be misleading on their own.

2. PERMUTATION IMPORTANCE on validation folds (training data only).
   Shuffle one feature's column in the validation fold and measure how much
   ROC-AUC drops. Works for any model. Correlated features hide each other
   (shuffle tenure and the model can still use TotalCharges), so correlated
   GROUPS are also shuffled together.

Neither view is causal. A large coefficient means the feature helps
PREDICT churn in this snapshot, not that changing it would change churn.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from churn_platform.config import RANDOM_SEED
from churn_platform.features import MODEL_FEATURES

# Features whose values for churned customers are measured AT churn time
# (tenure = months stayed before leaving; TotalCharges = final bill).
RETROSPECTIVE_FEATURES = {"tenure", "TotalCharges"}

INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                   "StreamingTV", "StreamingMovies"]
# Shuffled together because they carry overlapping information.
FEATURE_GROUPS = {
    "GROUP tenure + TotalCharges": ["tenure", "TotalCharges"],
    "GROUP charges (Monthly + Total)": ["MonthlyCharges", "TotalCharges"],
    "GROUP InternetService + 6 add-ons": ["InternetService", *INTERNET_ADDONS],
    "GROUP PhoneService + MultipleLines": ["PhoneService", "MultipleLines"],
}


def early_warning_status(feature: str) -> str:
    if feature in RETROSPECTIVE_FEATURES:
        return "retrospective (measured at churn for churners)"
    return "usable if recorded before the prediction date"


def logreg_coefficients(fitted_pipeline, X_train: pd.DataFrame) -> pd.DataFrame:
    """Coefficient table from a FITTED logreg pipeline (build_logreg)."""
    pre = fitted_pipeline.named_steps["preprocess"]
    coef = dict(zip(pre.get_feature_names_out(), fitted_pipeline.named_steps["model"].coef_[0]))
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    num_names = pre.named_transformers_["num"].get_feature_names_out()   # incl. missing indicator

    rows = []
    for name, sd in zip(num_names, scaler.scale_):
        c = coef[f"num__{name}"]
        is_indicator = name.startswith("missingindicator_")
        feature = name.replace("missingindicator_", "")
        rows.append({
            "feature": feature, "term": "missing (blank)" if is_indicator else "+1 SD",
            "reference": "not missing" if is_indicator else "-",
            "training_sd": None if is_indicator else float(sd),
            "log_odds": c, "odds_ratio": float(np.exp(c)), "kind": "numeric",
        })

    encoder = pre.named_transformers_["cat"]
    for feature, categories in zip(encoder.feature_names_in_, encoder.categories_):
        reference = X_train[feature].value_counts().idxmax()       # most common category
        ref_coef = coef[f"cat__{feature}_{reference}"]
        for category in categories:
            if category == reference:
                continue
            c = coef[f"cat__{feature}_{category}"] - ref_coef
            rows.append({"feature": feature, "term": str(category), "reference": str(reference),
                         "training_sd": None, "log_odds": c, "odds_ratio": float(np.exp(c)),
                         "kind": "categorical"})

    table = pd.DataFrame(rows)
    table["direction"] = np.where(table["log_odds"] > 0, "higher churn odds", "lower churn odds")
    table["early_warning"] = table["feature"].map(early_warning_status)
    table["abs_log_odds"] = table["log_odds"].abs()
    return table.sort_values("abs_log_odds", ascending=False, ignore_index=True)


def permutation_importance_cv(model, X: pd.DataFrame, y, folds, n_repeats: int = 5,
                              groups: dict | None = None, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Mean drop in validation ROC-AUC when a feature (or group) is shuffled.

    For each fold: fit on the training part, score the untouched validation
    part, then shuffle each feature's values across validation rows
    (n_repeats times) and re-score. The model is never refitted on shuffled data.
    """
    y = np.asarray(y)
    targets = {f: [f] for f in MODEL_FEATURES}
    targets.update(groups or {})
    rng = np.random.default_rng(seed)
    drops = {name: [] for name in targets}
    for train_idx, val_idx in folds.split(X, y):
        fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        base = roc_auc_score(y_val, fitted.predict_proba(X_val)[:, 1])
        for name, cols in targets.items():
            for _ in range(n_repeats):
                X_perm = X_val.copy()
                order = rng.permutation(len(X_val))
                # One shared shuffle for the whole group keeps its columns consistent with each other.
                X_perm[cols] = X_val[cols].iloc[order].to_numpy()
                drops[name].append(base - roc_auc_score(y_val, fitted.predict_proba(X_perm)[:, 1]))
    rows = [{"feature": name, "is_group": name.startswith("GROUP"),
             "auc_drop_mean": float(np.mean(d)), "auc_drop_std": float(np.std(d, ddof=1)),
             "early_warning": "mixed" if name.startswith("GROUP") else early_warning_status(name)}
            for name, d in drops.items()]
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False, ignore_index=True)
