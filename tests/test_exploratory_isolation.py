"""analyze_models --skip-test must write only to the exploratory folder (redirected to a
temporary directory here) and never touch historical Phase 4 outputs or test rows.

This runs the real Phase 4 analysis in training-only mode (about 20-60 s).
Full mode is never run by the test suite.
"""

import hashlib

import pytest

from churn_platform import analyze_models
from churn_platform.config import MODELS_DIR, RAW_DATA_FILE, REPORTS_DIR, SPLIT_FILE

pytestmark = pytest.mark.skipif(
    not (RAW_DATA_FILE.exists() and SPLIT_FILE.exists() and (REPORTS_DIR / "phase4" / "decisions.json").exists()),
    reason="needs raw data, split file and Phase 4 outputs")


def tree_hashes(*dirs):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in dirs for p in d.rglob("*") if p.is_file()}


def test_skip_test_writes_only_to_exploratory_dir(tmp_path, monkeypatch):
    from sklearn.pipeline import Pipeline

    from churn_platform.features import build_model_frame, split_data
    from churn_platform.validation import load_raw

    X, y, ids = build_model_frame(load_raw(RAW_DATA_FILE))
    _, X_test, *_ = split_data(X, y, ids)
    test_rows = set(X_test.index)

    touched_test_rows, inspected = [], {"frames": 0, "other": 0}
    original_fit, original_proba = Pipeline.fit, Pipeline.predict_proba

    def guard(rows):
        if not hasattr(rows, "index"):
            inspected["other"] += 1          # e.g. a NumPy array: no row identity to check
            return
        inspected["frames"] += 1
        if test_rows & set(rows.index):
            touched_test_rows.append(len(test_rows & set(rows.index)))

    def recording_fit(self, X, y=None, **kw):
        guard(X)
        return original_fit(self, X, y, **kw)

    def recording_proba(self, X, **kw):
        guard(X)
        return original_proba(self, X, **kw)

    monkeypatch.setattr(Pipeline, "fit", recording_fit)
    monkeypatch.setattr(Pipeline, "predict_proba", recording_proba)

    exploratory = tmp_path / "phase4_exploratory"
    monkeypatch.setattr(analyze_models, "EXPLORATORY_DIR", exploratory)
    protected = [d for d in REPORTS_DIR.iterdir() if d.is_dir()] + [MODELS_DIR]
    protected_files = [p for p in REPORTS_DIR.iterdir() if p.is_file()]
    before = tree_hashes(*protected) | {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_files}

    assert analyze_models.main(["--skip-test", "--permutation-repeats", "1"]) == 0

    after = tree_hashes(*protected) | {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected_files}
    assert after == before                                   # nothing under reports/ or models/ changed
    written = {p.relative_to(exploratory).as_posix() for p in exploratory.rglob("*") if p.is_file()}
    assert {"decisions.json", "phase4_model_analysis.md", "phase4_results.json",
            "threshold_cost_sensitivity.csv", "figures/phase4_threshold_cost.png"} <= written
    assert "final_test_evaluation.json" not in written
    assert touched_test_rows == []                           # no fit or prediction ever saw a test row
    assert inspected["frames"] > 100 and inspected["other"] == 0   # the guard really checked every call
