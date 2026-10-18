"""Tests for churn_platform.verify: passes on the real project, detects tampering,
writes nothing, and never fits models or scores dataset rows.

Tampering is always done on COPIES in pytest temporary directories.
"""

import hashlib
import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from churn_platform import verify
from churn_platform.config import ARTIFACT_FILE, PROJECT_ROOT, RAW_DATA_FILE, SPLIT_FILE
from churn_platform.verify import Paths, run_checks

pytestmark = pytest.mark.skipif(
    not (RAW_DATA_FILE.exists() and SPLIT_FILE.exists() and ARTIFACT_FILE.exists()),
    reason="needs raw data, split file and the Phase 5 artifact")


def failed(results):
    return {r["check"] for r in results if not r["passed"]}


def snapshot(root, skip=(".venv", ".pytest_cache")):
    """Hash of every file under root (including __pycache__), so any write is visible."""
    out = {}
    for p in root.rglob("*"):
        if p.is_file() and not any(part in skip for part in p.relative_to(root).parts):
            out[str(p.relative_to(root))] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    return out


def test_real_project_passes():
    results = run_checks()
    assert failed(results) == set(), results
    assert [r["check"] for r in results] == [name for name, _ in verify.CHECKS]


def test_command_writes_no_project_files_and_exits_zero():
    before = snapshot(PROJECT_ROOT)
    proc = subprocess.run([sys.executable, "-B", "-m", "churn_platform.verify"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True)
    after = snapshot(PROJECT_ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RESULT: PASSED" in proc.stdout
    assert after == before


def test_never_fits_and_only_predicts_the_single_api_example(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    import churn_platform.modeling as modeling

    def forbidden(*args, **kwargs):
        raise AssertionError("verify must not fit models or compute evaluation metrics")

    rows_scored = []
    original = Pipeline.predict_proba

    def recording(self, X, **kw):
        rows_scored.append(len(X))
        return original(self, X, **kw)

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(modeling, "evaluate", forbidden)
    monkeypatch.setattr(Pipeline, "predict_proba", recording)
    assert failed(run_checks()) == set()
    assert rows_scored == [1]          # exactly one made-up customer, via the API smoke test


# --- tamper detection (copies only) ------------------------------------------------

def test_detects_changed_historical_result(tmp_path):
    manifest = PROJECT_ROOT / "reports" / "historical_results.sha256"
    shutil.copy(manifest, tmp_path / manifest.name)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            rel = line.split(maxsplit=1)[1]
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(PROJECT_ROOT / rel, tmp_path / rel)
    paths = Paths(root=tmp_path, historical=tmp_path / manifest.name)
    assert failed(run_checks(paths)) == set()                       # faithful copy passes
    target = tmp_path / "reports" / "phase4" / "decisions.json"
    target.write_text(target.read_text(encoding="utf-8").replace("0.16", "0.20"), encoding="utf-8")
    assert failed(run_checks(paths)) == {"historical results"}


def test_detects_changed_raw_data(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_bytes(RAW_DATA_FILE.read_bytes() + b"\n")
    assert "raw data" in failed(run_checks(Paths(raw=raw)))


def test_detects_changed_or_missing_split_file(tmp_path):
    split = pd.read_csv(SPLIT_FILE)
    split.loc[split.index[split["split"] == "test"][0], "split"] = "train"
    tampered = tmp_path / "split.csv"
    split.to_csv(tampered, index=False)
    assert failed(run_checks(Paths(split=tampered))) >= {"split"}
    missing = run_checks(Paths(split=tmp_path / "missing.csv"))
    assert "make_split" in next(r["detail"] for r in missing if r["check"] == "split")


def test_detects_tampered_artifact(tmp_path):
    from churn_platform.artifact import metadata_path_for
    copy = tmp_path / ARTIFACT_FILE.name
    copy.write_bytes(ARTIFACT_FILE.read_bytes() + b"x")
    shutil.copy(metadata_path_for(ARTIFACT_FILE), metadata_path_for(copy))
    assert failed(run_checks(Paths(artifact=copy))) >= {"model artifact", "API smoke test"}


def test_detects_threshold_that_differs_from_phase4_decision(tmp_path):
    decisions = json.loads((PROJECT_ROOT / "reports" / "phase4" / "decisions.json").read_text(encoding="utf-8"))
    changed = tmp_path / "decisions.json"
    changed.write_text(json.dumps({**decisions, "selected_threshold": 0.2}), encoding="utf-8")
    assert "model artifact" in failed(run_checks(Paths(decisions=changed)))


def test_main_returns_nonzero_when_a_check_fails(monkeypatch, capsys):
    def always_fails(ctx):
        raise AssertionError("simulated failure")

    monkeypatch.setattr(verify, "CHECKS", verify.CHECKS + [("simulated", always_fails)])
    assert verify.main() == 1
    assert "FAILED (1 of 8 checks)" in capsys.readouterr().out
