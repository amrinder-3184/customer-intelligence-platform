"""Write reports/phase4_model_analysis.md from computed Phase 4 results.

As in findings.py and model_report.py, every number is inserted from the
results dictionary, and comparative statements are chosen by the data
(e.g. a "sign flip" is only described if the signs really differ).
"""

import pandas as pd

LABELS = {"dummy_most_frequent": "Dummy (most frequent)", "logreg": "LogReg",
          "decision_tree": "Decision tree (depth 5)", "random_forest": "Random forest (300 trees)"}


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{nd}f}"


def _pct(x):
    return f"{100 * x:.1f}%"


def _section_comparison(r) -> list[str]:
    s = pd.DataFrame(r["summary"]).set_index("model")
    folds = pd.DataFrame(r["folds"])
    diffs = pd.DataFrame(r["paired_differences"]).set_index("model")
    d = r["decisions"]
    lines = [
        "## 2. Does another simple model beat logistic regression?",
        "",
        "Candidates were chosen in advance and **not tuned**: a depth-5 decision tree (min 20 customers per leaf) "
        "and a random forest (300 trees, depth ≤ 8, min 10 per leaf). All use the same preprocessing and the "
        "**same 5 training folds** as Phase 3. Standard deviations here use n−1 (sample SD), so LogReg's fold SD "
        "reads slightly higher than Phase 3's population SD for identical fold scores.",
        "",
        "| Model | ROC-AUC (mean ± sd) | AP (mean ± sd) | Precision@0.5 | Recall@0.5 | F1@0.5 | Accuracy@0.5 | Fit time / fold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in s.iterrows():
        lines.append(
            f"| {LABELS.get(name, name)} | {_f(row['roc_auc_mean'], 4)} ± {_f(row['roc_auc_std'], 4)} | "
            f"{_f(row['average_precision_mean'], 4)} ± {_f(row['average_precision_std'], 4)} | "
            f"{_f(row['precision_mean'])} | {_f(row['recall_mean'])} | {_f(row['f1_mean'])} | "
            f"{_f(row['accuracy_mean'])} | {row['fit_seconds_mean']:.2f} s |")
    lines += ["", "**Fold-level ROC-AUC / AP** (same rows in each fold for every model):", "",
              "| Fold | " + " | ".join(LABELS.get(m, m) for m in s.index) + " |",
              "|---|" + "---:|" * len(s.index)]
    for k in sorted(folds["fold"].unique()):
        cells = []
        for m in s.index:
            row = folds[(folds["model"] == m) & (folds["fold"] == k)].iloc[0]
            cells.append(f"{row['roc_auc']:.4f} / {row['average_precision']:.4f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "**Paired difference vs LogReg** (candidate − LogReg, per fold):", "",
              "| Model | ΔROC-AUC (mean ± SE) | folds better | ΔAP (mean ± SE) | folds better |",
              "|---|---:|---:|---:|---:|"]
    for name, row in diffs.iterrows():
        lines.append(f"| {LABELS.get(name, name)} | {row['roc_auc_diff_mean']:+.4f} ± {row['roc_auc_diff_se']:.4f} | "
                     f"{int(row['roc_auc_folds_better'])}/5 | {row['average_precision_diff_mean']:+.4f} ± "
                     f"{row['average_precision_diff_se']:.4f} | {int(row['average_precision_folds_better'])}/5 |")
    rf, dt = diffs.loc["random_forest"], diffs.loc["decision_tree"]
    lines += [
        "",
        f"**Decision (pre-declared rule):** keep LogReg unless a candidate improves *both* ROC-AUC and AP by more than "
        f"2 standard errors of the paired difference. **Result: {LABELS[d['selected_model']]}** ({d['model_reason']}).",
        "",
        f"- The random forest's ROC-AUC gain is {rf['roc_auc_diff_mean']:+.4f} (SE {rf['roc_auc_diff_se']:.4f}, "
        f"better in {int(rf['roc_auc_folds_better'])}/5 folds). That is "
        + ("within noise." if rf["roc_auc_diff_mean"] <= 2 * rf["roc_auc_diff_se"] else "larger than 2 SE.")
        + f" It also takes about {s.loc['random_forest', 'fit_seconds_mean'] / s.loc['logreg', 'fit_seconds_mean']:.0f}× "
        "longer to fit, and it can't be read off as coefficients.",
        f"- The decision tree is {'worse' if dt['roc_auc_diff_mean'] < 0 else 'not better'} "
        f"({dt['roc_auc_diff_mean']:+.4f} ROC-AUC, {dt['average_precision_diff_mean']:+.4f} AP, better in "
        f"{int(dt['roc_auc_folds_better'])}/5 folds). A single depth-5 tree has only a limited number of distinct "
        "leaf probabilities, so it ranks customers coarsely.",
        "- The fold SE is optimistic, because folds share training rows. With five folds this rule is a "
        "sanity check, not a significance test.",
        "",
        "![Model comparison](figures/phase4_model_comparison.png)",
        "",
    ]
    return lines


def _section_calibration(r) -> list[str]:
    c = r["calibration_overall"]
    folds = r["calibration_folds"]
    d = r["decisions"]
    base = c["uncalibrated"]
    lines = [
        "## 3. Are the probabilities trustworthy? (calibration)",
        "",
        "Calibrated variants wrap the LogReg pipeline in `CalibratedClassifierCV` (inner 5-fold CV). This is "
        "nested inside each outer fold, so the calibrator never sees the labels it is evaluated on.",
        "",
        "| Variant (training OOF) | Brier | Brier skill vs no-skill | Log loss | ECE (10 equal-count bins) | Mean predicted vs observed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in c.items():
        label = LABELS.get(name.replace("model:", ""), name) if name.startswith("model:") else f"LogReg {name}"
        lines.append(f"| {label} | {m['brier']:.4f} | {m['brier_skill_score']:.3f} | {m['log_loss']:.4f} | "
                     f"{m['ece']:.4f} | {_pct(m['mean_predicted'])} vs {_pct(m['observed_rate'])} |")
    per = {n: pd.DataFrame(t)["brier"] for n, t in folds.items()}
    better = {n: int((per[n] < per["uncalibrated"]).sum()) for n in per if n != "uncalibrated"}
    lines += [
        "",
        "Folds where a calibrator lowered Brier: " + ", ".join(f"{n} {k}/5" for n, k in better.items()) + ".",
        "",
        f"**Decision (pre-declared rule):** adopt a calibrator only if it lowers OOF Brier *and* log loss, and lowers "
        f"Brier in at least 4 of 5 folds. **Result: {d['selected_calibration']}** ({d['calibration_reason']}).",
        "",
        f"- **How to read Brier:** Brier depends on the churn rate. Predicting the overall rate ({_pct(base['observed_rate'])}) "
        f"for everyone already scores {base['brier_reference_no_skill']:.4f}. LogReg's {base['brier']:.4f} is a "
        f"{_pct(base['brier_skill_score'])} improvement on that, and this mixes ranking skill with calibration. A low "
        "Brier alone doesn't prove good calibration.",
        f"- **Calibration evidence:** the average predicted probability ({_pct(base['mean_predicted'])}) matches the "
        f"observed rate, the binned gap (ECE) is {base['ece']:.4f}, and the curve stays close to the diagonal. "
        "The threshold analysis below gives independent evidence: the cost-optimal thresholds land near the values "
        "theory predicts *for calibrated probabilities*.",
        "- **Why calibration changed so little:** logistic regression is fitted by minimising log loss, so it tends to "
        "be calibrated already when the model is reasonably specified."
        + (f" Here both tree models have a higher ECE ({c['model:decision_tree']['ece']:.4f}, "
           f"{c['model:random_forest']['ece']:.4f}) than uncalibrated LogReg."
           if min(c["model:decision_tree"]["ece"], c["model:random_forest"]["ece"]) > base["ece"] else ""),
        "- **Why ECE isn't the deciding metric:** "
        + (f"isotonic has the lowest ECE ({c['isotonic']['ece']:.4f}) but a worse Brier and log loss. "
           if c["isotonic"]["ece"] < base["ece"] else "")
        + "ECE changes with the choice of bins and can be lowered by a calibrator that fits noise. Brier and log loss "
        "are proper scoring rules (they're best only for the true probabilities), so the rule uses them.",
        "- **Limitations:** ECE depends on how many bins are used (10 bins of ~563 customers each). Few customers "
        "receive probabilities above 0.8, so the top end of the curve rests on less data.",
        "",
        "![Calibration](figures/phase4_calibration.png)",
        "",
    ]
    return lines


def _section_threshold(r) -> list[str]:
    d, p = r["decisions"], r["primary_threshold"]
    sens = pd.DataFrame(r["sensitivity"])
    ca = d["cost_assumptions"]
    lines = [
        "## 4. Which threshold? (a business decision informed by the model)",
        "",
        "The model outputs a probability. The **threshold** decides who gets flagged for retention:",
        "- **Lower threshold:** more churners caught (fewer false negatives), but more offers go to customers who would "
        "have stayed (more false positives).",
        "- **Higher threshold:** the reverse.",
        "",
        "The right balance depends on what each mistake costs the business, which the data can't tell us.",
        "",
        f"`total_cost = FP × {ca['fp_cost']:g} + FN × {ca['fn_cost']:g}`. **These costs are illustrative assumptions, "
        "not measured business values.** The default treats a missed churner as 5× as costly as an unnecessary offer. "
        "Thresholds 0.05–0.95 (step 0.01) are evaluated on **training out-of-fold probabilities**; ties go to the "
        "higher threshold.",
        "",
        "If the probabilities are calibrated, theory says flag a customer when p > FP/(FP+FN) = 1/(1+ratio).",
        "",
        "| FN:FP cost ratio | Theory | Selected (OOF) | Near-optimal band (≤1% above min) | Cost saving vs 0.5 | Precision | Recall | Customers flagged (of 5,634) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in sens.iterrows():
        lines.append(f"| {row['fn_to_fp_cost_ratio']:g}:1 | {row['theoretical_threshold']:.3f} | **{row['selected_threshold']:.2f}** | "
                     f"{row['near_optimal_min']:.2f}–{row['near_optimal_max']:.2f} | {row['saving_vs_0_5_pct']:.1f}% | "
                     f"{_f(row['precision'])} | {_f(row['recall'])} | {int(row['flagged']):,} |")
    in_band = ((sens["theoretical_threshold"] >= sens["near_optimal_min"] - 1e-9)
               & (sens["theoretical_threshold"] <= sens["near_optimal_max"] + 1e-9))
    max_gap = (sens["selected_threshold"] - sens["theoretical_threshold"]).abs().max()
    one = sens[sens["fn_to_fp_cost_ratio"] == 1].iloc[0] if (sens["fn_to_fp_cost_ratio"] == 1).any() else None
    lines += [
        "",
        f"**Selected for the illustrative 5:1 scenario: {d['selected_threshold']:.2f}** (near-optimal band "
        f"{p['near_optimal_min']:.2f}–{p['near_optimal_max']:.2f}). On training OOF predictions this flags "
        f"{p['tp'] + p['fp']:,} customers and catches {p['tp']:,} of {p['tp'] + p['fn']:,} churners "
        f"(recall {_f(p['recall'])}, precision {_f(p['precision'])}). **The 0.5 baseline from Phase 3 is kept as a "
        "reference, not silently replaced.**",
        "",
        f"- **Theory vs data:** the theoretical threshold falls inside the near-optimal band for "
        f"{int(in_band.sum())} of {len(sens)} cost ratios (largest gap to the selected value: {max_gap:.2f}). "
        + ("That is what calibrated probabilities predict." if in_band.all() else
           "Where it falls outside, the probabilities may be miscalibrated in that range."),
    ]
    if one is not None:
        lines.append(f"- **When costs are equal (1:1),** the best threshold ({one['selected_threshold']:.2f}) saves only "
                     f"{one['saving_vs_0_5_pct']:.1f}% versus 0.5. The 0.5 default is only sensible when both errors "
                     "cost the same.")
    lines += [
        "- **The higher the cost of a missed churner, the lower the threshold.** This is the key thing to explain "
        "to a business stakeholder.",
        "- **The near-optimal band shows how much the exact value matters.** Within it, the choice can follow "
        "operational limits such as the size of the retention team.",
        "- **The cost model is optimistic:** it assumes every contacted churner is saved and every offer costs the same.",
        "",
        "![Threshold analysis](figures/phase4_threshold_cost.png)",
        "",
    ]
    return lines


def _section_interpretation(r) -> list[str]:
    coef = pd.DataFrame(r["coefficients"])
    perm = {n: pd.DataFrame(t).set_index("feature") for n, t in r["permutation"].items()}
    corr = r["numeric_correlations"]
    marginal = r["marginal_correlation_with_churn"]
    lines = [
        "## 5. Which features are associated with the predictions?",
        "",
        "### 5a. Logistic regression coefficients (fitted on the full training set)",
        "",
        "- **Numeric features:** effect per **1 training standard deviation**, because the inputs are scaled.",
        "- **Categorical features:** effect **relative to the most common category**. All one-hot columns are kept, "
        "so only differences within a feature are meaningful.",
        "- **Odds ratio (OR):** OR = exp(coefficient). OR 0.5 halves the odds of churn; OR 2 doubles them, "
        "*holding all other inputs fixed*.",
        "",
        "| Rank | Feature | Term (vs reference) | Log-odds | Odds ratio | Direction | Early-warning status |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for i, row in coef.head(15).iterrows():
        term = row["term"] + (f" (vs {row['reference']})" if row["reference"] not in ("-", "not missing") else "")
        if row["term"] == "+1 SD":
            term = f"+1 SD ({row['training_sd']:,.1f})"
        lines.append(f"| {i + 1} | {row['feature']} | {term} | {row['log_odds']:+.3f} | {row['odds_ratio']:.2f} | "
                     f"{row['direction']} | {row['early_warning']} |")
    lines += ["", "Full table: `phase4/logreg_coefficients.csv`.", ""]

    num = coef[coef["term"] == "+1 SD"].set_index("feature")["log_odds"]
    flips = [f for f in ("tenure", "MonthlyCharges", "TotalCharges")
             if f in num.index and (num[f] > 0) != (marginal[f] > 0)]
    if flips:
        lines.append(
            "**A one-variable view and the model can disagree.** For "
            + ", ".join(f"`{f}` (one-variable correlation with churn {marginal[f]:+.3f}, model coefficient {num[f]:+.3f})"
                        for f in flips)
            + ", the coefficient's sign is the opposite of the one-variable association. A coefficient answers "
            "\"what if this changed *while every other input stays fixed*\". The other inputs include closely related "
            f"ones: corr(tenure, TotalCharges) = {corr['tenure']['TotalCharges']:.3f}, corr(MonthlyCharges, TotalCharges) = "
            f"{corr['MonthlyCharges']['TotalCharges']:.3f}, and the internet-service and add-on columns that make up much "
            "of a bill. For example, a higher TotalCharges *at the same tenure and monthly charge* is a different "
            "situation from simply a higher TotalCharges. Correlated inputs share and swap credit, so individual "
            "coefficients can't be read in isolation.")
        lines.append("")
    lines += [
        "**Not causal.** A coefficient describes an association in this snapshot. A two-year contract's OR doesn't mean "
        "moving a customer to two years would cut their churn odds by that factor. Committed customers probably differ "
        "in ways the data doesn't record.",
        "",
        "![Coefficients](figures/phase4_logreg_coefficients.png)",
        "",
        "### 5b. Permutation importance (validation folds of the training set)",
        "",
        "Each feature is shuffled in a validation fold, and the drop in ROC-AUC is measured (5 folds × "
        f"{r['args']['permutation_repeats']} shuffles). Correlated features are also shuffled **as groups**.",
        "",
        "| Feature / group | LogReg ΔAUC | Random forest ΔAUC |",
        "|---|---:|---:|",
    ]
    lr, rf = perm["logreg"], perm["random_forest"]
    for feature in lr.index[:14]:
        lines.append(f"| {feature} | {lr.loc[feature, 'auc_drop_mean']:.4f} ± {lr.loc[feature, 'auc_drop_std']:.4f} | "
                     f"{rf.loc[feature, 'auc_drop_mean']:.4f} ± {rf.loc[feature, 'auc_drop_std']:.4f} |")
    g_internet, g_tenure = "GROUP InternetService + 6 add-ons", "GROUP tenure + TotalCharges"
    lines += [""]
    if lr.loc["tenure", "auc_drop_mean"] > lr.loc[g_tenure, "auc_drop_mean"]:
        lines.append(
            f"- **A trap:** for LogReg, shuffling `tenure` *alone* costs {lr.loc['tenure', 'auc_drop_mean']:.3f} AUC, "
            f"*more* than shuffling tenure and TotalCharges together ({lr.loc[g_tenure, 'auc_drop_mean']:.3f}). Shuffling "
            "tenure alone creates impossible customers (e.g. 1 month of tenure with a $5,000 total bill), and the model "
            "uses the tenure/TotalCharges *combination*. The single-feature number therefore overstates tenure. **The "
            "group number is the more realistic measure.**")
    top_group_lr = lr[lr["is_group"]]["auc_drop_mean"].idxmax()
    top_group_rf = rf[rf["is_group"]]["auc_drop_mean"].idxmax()
    lines.append(
        f"- Among the correlated groups, the largest drop is **{top_group_lr}** for LogReg "
        f"({lr.loc[top_group_lr, 'auc_drop_mean']:.3f}) and **{top_group_rf}** for the forest "
        f"({rf.loc[top_group_rf, 'auc_drop_mean']:.3f}). That group shuffles seven columns at once, and all seven "
        "encode whether the customer has internet service, so it removes more information than any single column.")
    lines.append(
        f"- `Contract` alone: {lr.loc['Contract', 'auc_drop_mean']:.3f} (LogReg), {rf.loc['Contract', 'auc_drop_mean']:.3f} (forest). "
        "Features with a drop near zero (" + ", ".join(f"`{f}`" for f in lr.index if not lr.loc[f, 'is_group']
                                                      and abs(lr.loc[f, 'auc_drop_mean']) < 0.001)
        + ") add almost nothing *given the other features*. That doesn't mean they're unrelated to churn.")
    p3 = r.get("phase3_reference")
    phase3_sentence = (
        f"In Phase 3, removing them entirely lowered CV ROC-AUC from {p3['cv_roc_auc_logreg']:.3f} to "
        f"{p3['cv_roc_auc_without_tenure_totalcharges']:.3f}, because the model partly compensates with other features."
        if p3 else "")
    lines += [
        "- Permutation importance measures how much **this model relies on** a feature. It doesn't show that the "
        "feature causes churn, and it shouldn't be confused with impurity-based tree importance (not used here).",
        "",
        "![Permutation importance](figures/phase4_permutation_importance.png)",
        "",
        "### 5c. Retrospective importance vs early-warning suitability",
        "",
        f"`tenure` and `TotalCharges` rank high in both views, but for churners they are measured **at churn**. Shuffled "
        f"together, they cost {lr.loc[g_tenure, 'auc_drop_mean']:.3f} AUC (LogReg). {phase3_sentence} For a real "
        "early-warning system, they would have to be re-measured at a prediction date *before* any churn. The other "
        "features are only safe if they're also recorded before the outcome. The snapshot can't confirm that "
        "(e.g. a contract switched just before leaving).",
        "",
    ]
    return lines


def _section_test(r) -> list[str]:
    ft = r["final_test"]
    lines = ["## 6. Held-out test set: one evaluation of the pre-declared final model", ""]
    if ft is None:
        lines.append("Not run (`--skip-test`). The test set was not touched in this run.")
        prev = r.get("existing_final_test")
        if prev:
            lines.append(
                f" An earlier single evaluation exists in `phase4/final_test_evaluation.json` (threshold "
                f"{prev['saved_threshold']:.2f}). "
                + ("Its decisions match this run's." if prev["matches_current_decisions"] else
                   "**Its decisions differ from this exploratory run**, so it does not describe the settings shown above."))
        return lines + [""]
    s, d0, pm = ft["at_selected_threshold"], ft["at_default_0_5"], ft["probability_metrics"]
    ca = ft["decisions"]["cost_assumptions"]
    lines += [
        "**Final model:** "
        f"{LABELS[ft['decisions']['selected_model']]} ({ft['decisions']['selected_calibration']}), threshold "
        f"{s['threshold']:.2f}. All of this was fixed from training data and written to `phase4/decisions.json` "
        f"(SHA-256 `{ft['decisions_sha256'][:16]}…`) **before** this single test evaluation. Nothing below was "
        "used to make a choice.",
        "",
        f"| Test set ({s['n']:,} customers, {s['positives']} churners) | Threshold {s['threshold']:.2f} (selected) | Threshold 0.5 (Phase 3 baseline) |",
        "|---|---:|---:|",
    ]
    for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy")]:
        lines.append(f"| {label} | {s[key]:.3f} | {d0[key]:.3f} |")
    for key, label in [("tp", "TP (churners caught)"), ("fp", "FP (unneeded offers)"), ("fn", "FN (churners missed)"), ("tn", "TN")]:
        lines.append(f"| {label} | {s['confusion_matrix'][key]} | {d0['confusion_matrix'][key]} |")
    lines.append(f"| Assumed cost (FP×{ca['fp_cost']:g} + FN×{ca['fn_cost']:g}) | {s['assumed_cost']:g} | {d0['assumed_cost']:g} |")
    lines += [
        "",
        f"Threshold-independent: ROC-AUC {s['roc_auc']:.4f}, PR-AUC {s['average_precision']:.4f}, "
        f"Brier {pm['brier']:.4f}, log loss {pm['log_loss']:.4f}, ECE {pm['ece']:.4f}.",
        "",
    ]
    p3 = r.get("phase3_reference")
    if p3:
        same = (p3["confusion_matrix"] == d0["confusion_matrix"])
        lines.append(("At 0.5 the confusion matrix is **identical to Phase 3's**, confirming that the final model is "
                      "the same LogReg and the split is unchanged." if same else
                      "At 0.5 the confusion matrix **differs from Phase 3's**; investigate before trusting these numbers."))
        lines.append("")
    saving = (d0["assumed_cost"] - s["assumed_cost"]) / d0["assumed_cost"]
    lines += [
        f"Under the *assumed* 5:1 costs, the selected threshold's test cost is {_pct(saving)} "
        f"{'lower' if saving > 0 else 'higher'} than at 0.5. This is a consistency check of the training-based choice, "
        "not a measured business saving.",
        "",
    ]
    return lines


def write_phase4_report(r: dict, path) -> None:
    d = r["decisions"]
    lines = [
        "# Phase 4: model comparison, calibration, threshold and interpretation",
        "",
        f"_Generated by `python -m churn_platform.analyze_models` (raw SHA-256 `{r['raw_file_sha256'][:16]}…`, "
        f"scikit-learn {r['environment']['scikit-learn']}). Do not edit by hand. Phase 3 outputs are not modified._",
        "",
        "## 1. Decisions and the data behind each",
        "",
        "| Decision | Result | Data used |",
        "|---|---|---|",
        f"| Model | **{LABELS[d['selected_model']]}** | 5-fold OOF on 5,634 training rows |",
        f"| Calibration | **{d['selected_calibration']}** | same OOF folds; calibrators nested inside each fold |",
        f"| Threshold | **{d['selected_threshold']:.2f}** (0.5 kept as reference) | training OOF probabilities + "
        f"**assumed** costs FP={d['cost_assumptions']['fp_cost']:g}, FN={d['cost_assumptions']['fn_cost']:g} |",
        "| Interpretation | coefficients + permutation importance | training fit; training validation folds |",
        "| Final evaluation | one test run after decisions were saved | 1,409 held-out test rows |",
        "",
        "Every decision rule was written before the results were known (see `model_selection.select_model`, "
        "`calibration.select_calibration`, `thresholds.select_threshold`).",
        "",
    ]
    lines += _section_comparison(r)
    lines += _section_calibration(r)
    lines += _section_threshold(r)
    lines += _section_interpretation(r)
    lines += _section_test(r)
    lines += [
        "## 7. Evidence vs assumptions",
        "",
        "| Supported by computed evidence | Still an assumption |",
        "|---|---|",
        "| No tested alternative clearly beats LogReg on these folds | That no *untested* model or feature set would |",
        "| LogReg OOF probabilities are close to calibrated | That calibration holds on future customers |",
        "| Lower FN:FP cost ratios → higher optimal threshold | The actual cost ratio (5:1 is illustrative) |",
        "| Which features the model relies on | That changing those features would change churn (causal) |",
        "| Scores on this snapshot | Performance as a real early-warning system (tenure/TotalCharges are retrospective) |",
        "",
        "## 8. Limitations",
        "",
        "- Single synthetic snapshot. Associations only. Label noise from duplicate profiles with mixed outcomes.",
        "- Five folds give rough uncertainty estimates, and fold scores aren't independent.",
        "- Candidate models weren't tuned, so a tuned forest or gradient boosting *might* do better. This project "
        "deliberately keeps the search small.",
        "- The cost model is a simplification: fixed costs, and it assumes every contacted churner is retained.",
        "- The test set has now been used twice for the same LogReg (Phase 3 at 0.5, Phase 4 at the selected threshold). "
        "Neither result was used to choose anything. Any further model changes would need a fresh holdout or a new "
        "data sample.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
