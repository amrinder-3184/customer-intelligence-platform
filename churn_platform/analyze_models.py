"""Phase 4: model comparison, calibration, cost-based threshold, interpretation.

Usage (from the project root):
    python -m churn_platform.analyze_models                 # full run, incl. the single test evaluation
    python -m churn_platform.analyze_models --skip-test     # training-only analysis, test set untouched
    python -m churn_platform.analyze_models --skip-test --fn-cost 3   # explore other cost assumptions

Data used for each decision (all of it from the 5,634 TRAINING rows):
  A. Model choice        - out-of-fold (OOF) metrics on the Phase 3 CV folds
  B. Calibration choice  - OOF Brier / log loss, calibrators nested inside each fold
  C. Threshold           - OOF probabilities of the chosen model + assumed costs
  D. Interpretation      - coefficients from a fit on the training set;
                           permutation importance on validation folds
Then all decisions are written to reports/phase4/decisions.json, and only
after that is the pre-declared final model scored ONCE on the test set.

Test-set lock: reports/phase4/final_test_evaluation.json stores a hash of
decisions.json. If decisions change after the test result exists, the run
refuses to score the test set again. Re-running with identical decisions
just reproduces the same evaluation.
"""

import argparse
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

from churn_platform import plots
from churn_platform.calibration import (
    CALIBRATION_METHODS,
    build_calibrated,
    curve_table,
    per_fold_probability_metrics,
    probability_metrics,
    select_calibration,
)
from churn_platform.config import FIGURES_DIR, RANDOM_SEED, RAW_DATA_FILE, REPORTS_DIR
from churn_platform.download import sha256_of
from churn_platform.features import assert_no_test_rows, build_model_frame, split_data
from churn_platform.interpretation import FEATURE_GROUPS, logreg_coefficients, permutation_importance_cv
from churn_platform.model_selection import (
    build_candidates,
    compare_models,
    make_folds,
    oof_evaluate,
    paired_differences,
    select_model,
    summarize_folds,
)
from churn_platform.modeling import THRESHOLD, evaluate
from churn_platform.phase4_report import write_phase4_report
from churn_platform.thresholds import (
    COST_RATIOS,
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    THRESHOLD_GRID,
    cost_sensitivity,
    select_threshold,
    theoretical_threshold,
    threshold_table,
)
from churn_platform.train_baseline import SPLIT_FILE, environment
from churn_platform.validation import load_raw, validate

PHASE4_DIR = REPORTS_DIR / "phase4"
DECISIONS_FILE = PHASE4_DIR / "decisions.json"
FINAL_TEST_FILE = PHASE4_DIR / "final_test_evaluation.json"
# --skip-test writes ONLY here, so exploring cost assumptions can never overwrite
# the historical Phase 4 results (which came from the single test evaluation).
EXPLORATORY_DIR = REPORTS_DIR / "phase4_exploratory"


def decisions_hash(decisions: dict) -> str:
    return hashlib.sha256(json.dumps(decisions, sort_keys=True).encode("utf-8")).hexdigest()


def check_test_lock(current_hash: str) -> str | None:
    """None if the test set may be scored; otherwise the reason it may not."""
    if not FINAL_TEST_FILE.exists():
        return None
    previous = json.loads(FINAL_TEST_FILE.read_text(encoding="utf-8")).get("decisions_sha256")
    if previous == current_hash:
        return None  # same decisions -> re-running reproduces the same evaluation
    return ("decisions changed after the test set was already scored with different decisions. "
            "Scoring again would turn the test set into a selection tool. Use --skip-test, or "
            f"consciously delete {FINAL_TEST_FILE.name} and document why.")


def phase3_reference() -> dict | None:
    """Read (never write) the Phase 3 numbers the report compares against."""
    path = REPORTS_DIR / "phase3" / "metrics.json"
    if not path.exists():
        return None
    p3 = json.loads(path.read_text(encoding="utf-8"))
    cv = {row["model"]: row for row in p3["cv_results"]}
    return {
        "confusion_matrix": p3["test_metrics"]["logreg"]["confusion_matrix"],
        "cv_roc_auc_logreg": cv["logreg"]["roc_auc_mean"],
        "cv_roc_auc_without_tenure_totalcharges": cv["sensitivity_logreg_without_tenure_totalcharges"]["roc_auc_mean"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4 model analysis")
    parser.add_argument("--skip-test", action="store_true", help="do not touch the held-out test set")
    parser.add_argument("--fp-cost", type=float, default=DEFAULT_FP_COST, help="assumed cost of a false positive")
    parser.add_argument("--fn-cost", type=float, default=DEFAULT_FN_COST, help="assumed cost of a false negative")
    parser.add_argument("--permutation-repeats", type=int, default=5)
    args = parser.parse_args(argv)

    warnings.simplefilter("error", ConvergenceWarning)

    # Where outputs go. Full mode: the historical locations, exactly as before.
    if args.skip_test:
        out_dir, fig_dir = EXPLORATORY_DIR, EXPLORATORY_DIR / "figures"
        report_path, decisions_path = EXPLORATORY_DIR / "phase4_model_analysis.md", EXPLORATORY_DIR / "decisions.json"
    else:
        out_dir, fig_dir = PHASE4_DIR, FIGURES_DIR
        report_path, decisions_path = REPORTS_DIR / "phase4_model_analysis.md", DECISIONS_FILE
    warnings.simplefilter("ignore", UndefinedMetricWarning)  # Dummy's 0/0 precision; reported explicitly

    sha_before = sha256_of(RAW_DATA_FILE)
    raw = load_raw(RAW_DATA_FILE)
    if any(i.severity == "ERROR" for i in validate(raw)):
        print("Validation failed; run python -m churn_platform.inspect_data")
        return 1

    X, y, ids = build_model_frame(raw)
    X_train, X_test, y_train, y_test, id_train, id_test = split_data(X, y, ids)
    # Same split as Phase 3? Compare with the saved file if it exists.
    if SPLIT_FILE.exists():
        saved = pd.read_csv(SPLIT_FILE)
        if set(saved.loc[saved["split"] == "test", "customerID"]) != set(id_test):
            print("Split differs from data/processed/train_test_split.csv; aborting.")
            return 1
    test_index = X_test.index
    folds = make_folds()

    # ---- A. Model comparison (training only) -------------------------------
    assert_no_test_rows(X_train, test_index)
    print("A. Comparing models with 5-fold OOF predictions on the training set...")
    candidates = build_candidates()
    folds_table, oof = compare_models(candidates, X_train, y_train, folds)
    summary = summarize_folds(folds_table)
    diffs = paired_differences(folds_table)
    selected_model, model_reason = select_model(diffs)

    # ---- B. Calibration of the selected model (training only) ---------------
    assert_no_test_rows(X_train, test_index)
    print(f"B. Calibrating {selected_model} (nested CV)...")
    variants = {"uncalibrated": oof[selected_model]}
    for method in CALIBRATION_METHODS:
        variants[method], _ = oof_evaluate(build_calibrated(candidates[selected_model], method), X_train, y_train, folds)
    calib_overall = {name: probability_metrics(y_train, p) for name, p in variants.items()}
    calib_folds = {name: per_fold_probability_metrics(y_train, p, folds, X_train) for name, p in variants.items()}
    selected_calibration, calibration_reason = select_calibration(calib_folds, calib_overall)
    curves = pd.concat([curve_table(y_train, p, name) for name, p in variants.items()]
                       + [curve_table(y_train, oof[m], f"model:{m}") for m in ("decision_tree", "random_forest")],
                       ignore_index=True)
    for name in ("decision_tree", "random_forest"):
        calib_overall[f"model:{name}"] = probability_metrics(y_train, oof[name])

    # ---- C. Threshold from OOF probabilities of the chosen variant ----------
    final_oof = variants[selected_calibration]
    print("C. Cost-based threshold analysis on OOF probabilities...")
    primary_table = threshold_table(y_train, final_oof, args.fp_cost, args.fn_cost)
    primary = select_threshold(primary_table)
    sensitivity = cost_sensitivity(y_train, final_oof, fp_cost=args.fp_cost)

    # ---- D. Interpretation (training only) -----------------------------------
    assert_no_test_rows(X_train, test_index)
    print("D. Interpretation: coefficients and permutation importance...")
    logreg_fit = candidates["logreg"].fit(X_train, y_train)  # full training set; no test rows
    coefficients = logreg_coefficients(logreg_fit, X_train)
    perm = {name: permutation_importance_cv(candidates[name], X_train, y_train, folds,
                                            n_repeats=args.permutation_repeats, groups=FEATURE_GROUPS)
            for name in ("logreg", "random_forest")}
    numeric_corr = X_train[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
    # Marginal (one-variable) association with churn, to contrast with the
    # conditional (all-else-fixed) logistic coefficients.
    marginal_corr = {c: float(X_train[c].corr(y_train)) for c in ("tenure", "MonthlyCharges", "TotalCharges")}

    # ---- Decisions file (written BEFORE any test-set use) --------------------
    decisions = {
        "data_used": "training set only (5-fold stratified OOF, seed 42); test set not used",
        "selected_model": selected_model, "model_reason": model_reason,
        "selected_calibration": selected_calibration, "calibration_reason": calibration_reason,
        "cost_assumptions": {"fp_cost": args.fp_cost, "fn_cost": args.fn_cost,
                             "status": "ILLUSTRATIVE ASSUMPTION, not measured business costs"},
        "selected_threshold": primary["threshold"],
        "threshold_rule": "lowest total OOF cost on grid 0.05-0.95 step 0.01; ties -> higher threshold",
        "theoretical_threshold_if_calibrated": theoretical_threshold(args.fp_cost, args.fn_cost),
        "baseline_threshold_kept_for_reference": THRESHOLD,
        "seed": RANDOM_SEED,
    }
    current_hash = decisions_hash(decisions)
    # Check the lock BEFORE writing anything, so a refused run leaves every file unchanged.
    if not args.skip_test:
        blocked = check_test_lock(current_hash)
        if blocked:
            print(f"E. REFUSED: {blocked}")
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(json.dumps(decisions, indent=2), encoding="utf-8")

    # ---- E. Single test evaluation of the pre-declared final candidate -------
    final_test = None
    existing_test = None
    if args.skip_test:
        print("E. Skipped (--skip-test): the held-out test set was not touched.")
        if FINAL_TEST_FILE.exists():
            saved = json.loads(FINAL_TEST_FILE.read_text(encoding="utf-8"))
            existing_test = {"matches_current_decisions": saved.get("decisions_sha256") == current_hash,
                             "saved_threshold": saved["decisions"]["selected_threshold"]}
    else:
        print("E. Scoring the pre-declared final model once on the test set...")
        final_model = candidates[selected_model] if selected_calibration == "uncalibrated" \
            else build_calibrated(candidates[selected_model], selected_calibration)
        final_model.fit(X_train, y_train)
        test_proba = final_model.predict_proba(X_test)[:, 1]
        at_selected = evaluate(final_model, X_test, y_test, threshold=primary["threshold"])
        at_default = evaluate(final_model, X_test, y_test, threshold=THRESHOLD)
        for m in (at_selected, at_default):
            cm = m["confusion_matrix"]
            m["assumed_cost"] = cm["fp"] * args.fp_cost + cm["fn"] * args.fn_cost
            m.pop("classification_report")
        final_test = {
            "decisions_sha256": current_hash, "decisions": decisions,
            "note": "Scored once after all decisions were fixed from training data. Not used for any choice.",
            "at_selected_threshold": at_selected, "at_default_0_5": at_default,
            "probability_metrics": probability_metrics(y_test.to_numpy(), test_proba),
        }
        FINAL_TEST_FILE.write_text(json.dumps(final_test, indent=2), encoding="utf-8")

    # ---- Outputs ---------------------------------------------------------------
    folds_table.to_csv(out_dir / "model_comparison_folds.csv", index=False)
    summary.to_csv(out_dir / "model_comparison_summary.csv", index=False)
    diffs.to_csv(out_dir / "model_comparison_paired_differences.csv", index=False)
    primary_table.to_csv(out_dir / "threshold_table_primary.csv", index=False)
    sensitivity.to_csv(out_dir / "threshold_cost_sensitivity.csv", index=False)
    curves.to_csv(out_dir / "calibration_curves.csv", index=False)
    pd.concat([t.assign(variant=n) for n, t in calib_folds.items()]).to_csv(out_dir / "calibration_folds.csv", index=False)
    coefficients.to_csv(out_dir / "logreg_coefficients.csv", index=False)
    for name, table in perm.items():
        table.to_csv(out_dir / f"permutation_importance_{name}.csv", index=False)

    fig_dir.mkdir(parents=True, exist_ok=True)
    plots.plot_model_comparison(folds_table, fig_dir / "phase4_model_comparison.png")
    sensitivity_tables = {r: threshold_table(y_train, final_oof, args.fp_cost, args.fp_cost * r) for r in COST_RATIOS}
    plots.plot_threshold_costs(sensitivity_tables, sensitivity, primary_table, primary["threshold"],
                               fig_dir / "phase4_threshold_cost.png")
    plots.plot_calibration(curves, variants, fig_dir / "phase4_calibration.png")
    plots.plot_coefficients(coefficients, fig_dir / "phase4_logreg_coefficients.png")
    plots.plot_permutation_importance(perm, fig_dir / "phase4_permutation_importance.png")

    results = {
        "raw_file_sha256": sha_before, "environment": environment(), "args": vars(args),
        "summary": summary.to_dict(orient="records"), "folds": folds_table.to_dict(orient="records"),
        "paired_differences": diffs.to_dict(orient="records"), "decisions": decisions,
        "calibration_overall": calib_overall,
        "calibration_folds": {n: t.to_dict(orient="records") for n, t in calib_folds.items()},
        "primary_threshold": primary, "sensitivity": sensitivity.to_dict(orient="records"),
        "threshold_grid": [float(THRESHOLD_GRID[0]), float(THRESHOLD_GRID[-1])],
        "coefficients": coefficients.to_dict(orient="records"),
        "permutation": {n: t.to_dict(orient="records") for n, t in perm.items()},
        "numeric_correlations": numeric_corr.round(4).to_dict(),
        "marginal_correlation_with_churn": marginal_corr,
        "phase3_reference": phase3_reference(),
        "final_test": final_test, "existing_final_test": existing_test,
    }
    (out_dir / "phase4_results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    write_phase4_report(results, report_path)

    # ---- Console summary --------------------------------------------------------
    print("\nModel comparison (training OOF, mean +/- sd over 5 folds):")
    for _, r in summary.iterrows():
        print(f"  {r['model']:20} ROC-AUC {r['roc_auc_mean']:.4f}+/-{r['roc_auc_std']:.4f}  "
              f"AP {r['average_precision_mean']:.4f}+/-{r['average_precision_std']:.4f}  fit {r['fit_seconds_mean']:.2f}s")
    print(f"  -> selected: {selected_model} ({model_reason})")
    print("Calibration (training OOF):")
    for name in variants:
        m = calib_overall[name]
        print(f"  {name:13} Brier {m['brier']:.4f}  log loss {m['log_loss']:.4f}  ECE {m['ece']:.4f}")
    print(f"  -> selected: {selected_calibration} ({calibration_reason})")
    print(f"Threshold (FP cost {args.fp_cost:g}, FN cost {args.fn_cost:g}, ASSUMED): {primary['threshold']:.2f} "
          f"(near-optimal {primary['near_optimal_min']:.2f}-{primary['near_optimal_max']:.2f}; "
          f"theory {theoretical_threshold(args.fp_cost, args.fn_cost):.3f})")
    for _, r in sensitivity.iterrows():
        print(f"  ratio {r['fn_to_fp_cost_ratio']:>2}: threshold {r['selected_threshold']:.2f} "
              f"(theory {r['theoretical_threshold']:.3f}), cost saving vs 0.5: {r['saving_vs_0_5_pct']:.1f}%")
    if final_test:
        s, d = final_test["at_selected_threshold"], final_test["at_default_0_5"]
        print(f"Test (once): threshold {s['threshold']:.2f} -> precision {s['precision']:.3f} recall {s['recall']:.3f} "
              f"cost {s['assumed_cost']:g} | at 0.5 -> precision {d['precision']:.3f} recall {d['recall']:.3f} cost {d['assumed_cost']:g}")
    raw_unchanged = sha256_of(RAW_DATA_FILE) == sha_before
    print(f"Report: {report_path}")
    print(f"Raw CSV unchanged: {raw_unchanged}")
    print("RESULT:", "PASSED" if raw_unchanged else "FAILED")
    return 0 if raw_unchanged else 1


if __name__ == "__main__":
    sys.exit(main())
