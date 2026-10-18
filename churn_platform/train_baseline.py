"""Phase 3: leakage-safe baseline and logistic-regression evaluation.

Usage (from the project root):
    python -m churn_platform.train_baseline

Order of operations (this order is the leakage protection):
  1. Validate raw data (Phase 1 gate).
  2. Stateless typing -> X, y. Stratified 80/20 split with seed 42.
     The split is saved to data/processed/train_test_split.csv for auditing.
  3. Everything that involves CHOOSING (imputation strategy, class weights,
     sensitivity checks) uses 5-fold cross-validation on the TRAINING set only.
  4. The pre-declared models are fitted on the full training set and scored
     ONCE on the test set at the fixed 0.5 threshold. No result from step 4
     feeds back into any decision.
  5. Metrics -> reports/phase3/metrics.json, cv_results.csv;
     charts -> reports/figures/phase3_*.png; report -> reports/phase3_model_evaluation.md

No model file is saved; persistence is Phase 5.
"""

import json
import platform
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from churn_platform import plots
from churn_platform.config import FIGURES_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, RAW_DATA_FILE, REPORTS_DIR
from churn_platform.download import sha256_of
from churn_platform.features import NUMERIC_FEATURES, build_model_frame, profile_overlap, split_data
from churn_platform.model_report import write_model_report
from churn_platform.modeling import (
    IMPUTATION_STRATEGIES,
    THRESHOLD,
    build_logreg,
    build_models,
    cross_validate_models,
    evaluate,
)
from churn_platform.validation import load_raw, validate

PHASE3_DIR = REPORTS_DIR / "phase3"
SPLIT_FILE = PROCESSED_DATA_DIR / "train_test_split.csv"


def environment() -> dict:
    import matplotlib
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
    }


def blank_total_charges_rows(raw, X, y, ids, test_ids) -> list[dict]:
    mask = X["TotalCharges"].isna()
    return [
        {"customerID": cid, "tenure": int(t), "MonthlyCharges": float(m), "Contract": c,
         "churned": int(label), "split": "test" if cid in test_ids else "train"}
        for cid, t, m, c, label in zip(ids[mask], X.loc[mask, "tenure"], X.loc[mask, "MonthlyCharges"],
                                       X.loc[mask, "Contract"], y[mask])
    ]


def main() -> int:
    # Any convergence problem must stop the run, not scroll past as a warning.
    warnings.simplefilter("error", ConvergenceWarning)
    # Expected: the Dummy model never predicts churn, so precision is 0/0.
    # evaluate() records this as precision_defined=False and the report says so.
    warnings.simplefilter("ignore", UndefinedMetricWarning)

    sha_before = sha256_of(RAW_DATA_FILE)
    raw = load_raw(RAW_DATA_FILE)
    if any(i.severity == "ERROR" for i in validate(raw)):
        print("Validation failed; run python -m churn_platform.inspect_data")
        return 1

    # --- 1. Problem frame and split ---------------------------------------
    X, y, ids = build_model_frame(raw)
    X_train, X_test, y_train, y_test, id_train, id_test = split_data(X, y, ids)
    assert set(id_train).isdisjoint(id_test), "a customer is in both train and test"

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([
        pd.DataFrame({"customerID": id_train, "split": "train"}),
        pd.DataFrame({"customerID": id_test, "split": "test"}),
    ]).to_csv(SPLIT_FILE, index=False)

    split_info = {
        "seed": RANDOM_SEED, "test_size": len(X_test) / len(X),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
        "features": list(X.columns), "n_features_raw": X.shape[1],
        "customer_ids_disjoint": True,
    }
    blank_rows = blank_total_charges_rows(raw, X, y, ids, set(id_test))
    overlap = profile_overlap(X_train, y_train, X_test, y_test)
    overlap_mask = overlap.pop("overlap_mask")

    # --- 2. Cross-validation on TRAINING data only -------------------------
    cv_models = build_models("median_indicator")
    cv_models["logreg_zero_fill"] = build_logreg("zero")
    cv_models["sensitivity_logreg_without_tenure_totalcharges"] = build_logreg(
        numeric_features=[c for c in NUMERIC_FEATURES if c not in ("tenure", "TotalCharges")])
    cv = cross_validate_models(cv_models, X_train, y_train)

    # --- 3. Fit on full training set, evaluate ONCE on test ----------------
    models = build_models("median_indicator")
    test_metrics, train_metrics, fitted = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        train_metrics[name] = evaluate(model, X_train, y_train)
        test_metrics[name] = evaluate(model, X_test, y_test)

    # Sensitivity: does the small train/test profile overlap flatter the score?
    no_overlap = ~overlap_mask
    overlap_sensitivity = {
        name: evaluate(fitted[name], X_test[no_overlap], y_test[no_overlap])
        for name in ("logreg", "logreg_balanced")
    }
    logreg = fitted["logreg"]
    preprocess = logreg.named_steps["preprocess"]
    pipeline_info = {
        "n_features_after_preprocessing": int(len(preprocess.get_feature_names_out())),
        "feature_names_after_preprocessing": preprocess.get_feature_names_out().tolist(),
        "train_median_TotalCharges_used_for_imputation": float(
            preprocess.named_transformers_["num"].named_steps["impute"].statistics_[NUMERIC_FEATURES.index("TotalCharges")]),
        "logreg_iterations": int(logreg.named_steps["model"].n_iter_[0]),
    }

    # --- 4. Outputs ----------------------------------------------------------
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cv.to_csv(PHASE3_DIR / "cv_results.csv", index=False)

    probas = {name: m.predict_proba(X_test)[:, 1] for name, m in fitted.items()}
    plots.plot_confusion_matrices(test_metrics, FIGURES_DIR / "phase3_confusion_matrices.png")
    plots.plot_roc_pr_curves(y_test.to_numpy(), probas, FIGURES_DIR / "phase3_roc_pr_curves.png")

    results = {
        "raw_file_sha256": sha_before, "environment": environment(), "threshold": THRESHOLD,
        "split": split_info, "blank_total_charges_rows": blank_rows,
        "train_test_profile_overlap": overlap, "pipeline": pipeline_info,
        "cv_results": cv.to_dict(orient="records"),
        "test_metrics": test_metrics, "train_metrics": train_metrics,
        "test_metrics_excluding_overlap_rows": overlap_sensitivity,
    }
    (PHASE3_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_model_report(results, REPORTS_DIR / "phase3_model_evaluation.md")

    # --- Console summary ------------------------------------------------------
    print(f"Split: train {len(X_train)} (churn {y_train.mean():.4f}) | test {len(X_test)} (churn {y_test.mean():.4f})")
    print(f"Blank TotalCharges: {sum(r['split'] == 'train' for r in blank_rows)} train, "
          f"{sum(r['split'] == 'test' for r in blank_rows)} test")
    print(f"Test rows whose profile also appears in train: {overlap['test_rows_with_profile_in_train']}")
    print("\n5-fold CV on training set (mean +/- std):")
    for _, r in cv.iterrows():
        print(f"  {r['model']:48} ROC-AUC {r['roc_auc_mean']:.4f}+/-{r['roc_auc_std']:.4f}  "
              f"PR-AUC {r['average_precision_mean']:.4f}  recall {r['recall_mean']:.4f}  F1 {r['f1_mean']:.4f}")
    print(f"\nTest set (threshold {THRESHOLD}):")
    for name, m in test_metrics.items():
        cm = m["confusion_matrix"]
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.4f}"
        print(f"  {name:20} acc {m['accuracy']:.4f}  prec {m['precision']:.4f}  rec {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  ROC-AUC {auc}  PR-AUC {m['average_precision']:.4f}  "
              f"TN {cm['tn']} FP {cm['fp']} FN {cm['fn']} TP {cm['tp']}")
    raw_unchanged = sha256_of(RAW_DATA_FILE) == sha_before
    print(f"\nReport: {REPORTS_DIR / 'phase3_model_evaluation.md'}")
    print(f"Raw CSV unchanged: {raw_unchanged}")
    print("RESULT:", "PASSED" if raw_unchanged else "FAILED")
    return 0 if raw_unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
