"""The README's results table must match the saved result files exactly.

Values are READ from the saved reports; nothing is recomputed, and no model is
fitted or scored. If a report changes, or someone edits a README number by
hand, this test fails.
"""

import json
import re
import sys

import pandas as pd
import pytest

from churn_platform.config import PROJECT_ROOT, REPORTS_DIR

README = PROJECT_ROOT / "README.md"


def _json(rel):
    return json.loads((REPORTS_DIR / rel).read_text(encoding="utf-8"))


def _summary(model, column):
    return pd.read_csv(REPORTS_DIR / "phase4" / "model_comparison_summary.csv").set_index("model").loc[model, column]


def _p3(model, key):
    return _json("phase3/metrics.json")["test_metrics"][model][key]


def _p4_test(key):
    return _json("phase4/final_test_evaluation.json")["at_selected_threshold"][key]


# README label -> (function that reads the saved value, number of decimals; None = integer)
EXPECTED = {
    "Churn rate, all 7,043 customers": (lambda: _json("data_validation.json")["summary"]["churn_rate"], 4),
    'Phase 3 test, Dummy (always "no churn") accuracy': (lambda: _p3("dummy_most_frequent", "accuracy"), 3),
    "Phase 3 test, LogReg ROC-AUC": (lambda: _p3("logreg", "roc_auc"), 3),
    "Phase 3 test, LogReg PR-AUC": (lambda: _p3("logreg", "average_precision"), 3),
    "Phase 3 test, LogReg precision at 0.5": (lambda: _p3("logreg", "precision"), 3),
    "Phase 3 test, LogReg recall at 0.5": (lambda: _p3("logreg", "recall"), 3),
    "Phase 4 training CV, LogReg ROC-AUC": (lambda: _summary("logreg", "roc_auc_mean"), 3),
    "Phase 4 training CV, random forest ROC-AUC": (lambda: _summary("random_forest", "roc_auc_mean"), 3),
    "Phase 4 training CV, decision tree ROC-AUC": (lambda: _summary("decision_tree", "roc_auc_mean"), 3),
    "Phase 4 training OOF calibration error (ECE), LogReg":
        (lambda: _json("phase4/phase4_results.json")["calibration_overall"]["uncalibrated"]["ece"], 3),
    "Selected threshold (assumed 5:1 cost ratio)": (lambda: _json("phase4/decisions.json")["selected_threshold"], 2),
    "Phase 4 test at 0.16, precision": (lambda: _p4_test("precision"), 3),
    "Phase 4 test at 0.16, recall": (lambda: _p4_test("recall"), 3),
    "Phase 4 test at 0.16, churners caught (TP)": (lambda: _p4_test("confusion_matrix")["tp"], None),
    "Phase 4 test at 0.16, unnecessary offers (FP)": (lambda: _p4_test("confusion_matrix")["fp"], None),
    "Phase 4 test at 0.16, churners missed (FN)": (lambda: _p4_test("confusion_matrix")["fn"], None),
}


def readme_results_table() -> dict:
    text = README.read_text(encoding="utf-8")
    block = re.search(r"<!-- results-table:start -->(.*?)<!-- results-table:end -->", text, re.S)
    assert block, "results table markers missing from README"
    rows = {}
    for line in block.group(1).strip().splitlines()[2:]:          # skip header + separator
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1]
    return rows


def test_every_documented_result_is_checked_and_vice_versa():
    assert set(readme_results_table()) == set(EXPECTED)


@pytest.mark.parametrize("label", list(EXPECTED))
def test_documented_value_matches_saved_result(label):
    reader, decimals = EXPECTED[label]
    value = reader()
    expected = str(int(value)) if decimals is None else f"{value:.{decimals}f}"
    assert readme_results_table()[label] == expected


def test_parser_detects_a_wrong_number(monkeypatch, tmp_path):
    """Guards against a parser that always 'agrees': an edited copy must read back the edited value."""
    fake = tmp_path / "README.md"
    fake.write_text(README.read_text(encoding="utf-8").replace("| 0.842 |", "| 0.900 |", 1), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "README", fake)
    assert readme_results_table()["Phase 3 test, LogReg ROC-AUC"] == "0.900"


def test_readme_documents_safe_and_historical_commands_separately():
    text = README.read_text(encoding="utf-8")
    safe = text.split("## Reproduce from a fresh clone")[1].split("### Historical evaluation commands")[0]
    assert "make_split" in safe and "verify" in safe
    assert "train_baseline" not in safe and "analyze_models" not in safe   # no test-scoring command in the safe list
