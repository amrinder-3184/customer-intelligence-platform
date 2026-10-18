"""Prediction-problem definition: features, target, and the train/test split.

Problem: given one customer's attributes in the snapshot, estimate the
probability that Churn == "Yes".

IMPORTANT CAVEAT: this is a cross-sectional (one point in time) exercise.
Every column was recorded at the same moment as the outcome. For a customer
who already left, tenure means "months stayed before leaving" and
TotalCharges is the final bill, so they partly describe the churn itself. A
real early-warning model would have to take features at a prediction date
and observe churn in a LATER window. Results here measure how well the
features separate churners from non-churners in this snapshot, not how
well a deployed model would predict future churn.

Everything in this module is STATELESS (nothing is learned from the data),
so it is safe to apply before splitting. Anything learned from data
(medians, scaling, category lists) lives in the scikit-learn pipeline in
modeling.py and is fitted on the training set only.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_platform.analysis import CATEGORICAL_FEATURES, to_analysis_frame
from churn_platform.config import RANDOM_SEED
from churn_platform.validation import ID_COLUMN, NUMERIC_COLUMNS

NUMERIC_FEATURES = list(NUMERIC_COLUMNS)              # tenure, MonthlyCharges, TotalCharges
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 3 numeric + 16 categorical = 19
TEST_SIZE = 0.2

# Deliberately NOT used as predictors:
#   customerID           - an identifier; any pattern in it would be coincidence
#   Churn / churned      - the target
#   TotalCharges_blank   - the pipeline creates its own missing-value indicator
#   tenure_group, monthly_charges_band - Phase 2 display bins; the model sees raw values


def build_model_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Raw text frame -> (X, y, ids). Stateless; the raw frame is not modified.

    X:   the 19 predictor columns, typed (TotalCharges blank -> NaN, not 0).
    y:   1 if Churn == "Yes" else 0.
    ids: customerID, kept alongside for auditing, never given to a model.
    """
    df = to_analysis_frame(raw)
    X = df[MODEL_FEATURES].copy()
    y = df["churned"].rename("churned")
    ids = df[ID_COLUMN]
    return X, y, ids


def split_data(X: pd.DataFrame, y: pd.Series, ids: pd.Series, seed: int = RANDOM_SEED):
    """Stratified 80/20 split. Stratifying keeps the churn rate equal in both parts."""
    return train_test_split(X, y, ids, test_size=TEST_SIZE, stratify=y, random_state=seed)


def profile_overlap(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """How many test rows have a feature profile identical to some training row?

    These are different customers (customerIDs are unique), so this is not
    "the same customer in train and test" leakage. But a model can simply
    memorise such profiles, so we report how many exist and whether their
    labels agree. Nothing is deleted.
    """
    def keys(X):
        # One 64-bit hash per row of feature values (index ignored). NaN hashes
        # consistently, so two blank-TotalCharges rows can still match.
        return pd.util.hash_pandas_object(X[MODEL_FEATURES], index=False)

    train_keys, test_keys = keys(X_train), keys(X_test)
    in_train = test_keys.isin(set(train_keys))

    train_labels = pd.DataFrame({"key": train_keys.values, "y": y_train.values}).groupby("key")["y"].agg(set)
    agree = conflict = 0
    for key, label in zip(test_keys[in_train], y_test[in_train]):
        if train_labels[key] == {label}:
            agree += 1
        else:
            conflict += 1
    return {
        "test_rows": int(len(X_test)),
        "test_rows_with_profile_in_train": int(in_train.sum()),
        "share_of_test": float(in_train.mean()),
        "of_which_all_train_labels_agree": agree,
        "of_which_train_labels_differ_or_mixed": conflict,
        "overlap_mask": in_train.to_numpy(),
    }


def assert_no_test_rows(X: pd.DataFrame, test_index) -> None:
    """Guard for every Phase 4 selection step: refuse data that contains test rows.

    The split keeps the original row index, so overlap with the test index
    means test data has leaked into a decision.
    """
    leaked = X.index.intersection(test_index)
    if len(leaked):
        raise ValueError(f"{len(leaked)} held-out test rows passed to a training-only step")
