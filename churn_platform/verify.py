"""Phase 6: one read-only command that checks the project's integrity and wiring.

Usage (from the project root):
    python -B -m churn_platform.verify      # -B: don't write .pyc files either

Checks (each one independent; all must pass for exit code 0):
  1. raw data         - SHA-256 equals the committed fingerprint
  2. validation       - Phase 1 validator, run in memory (its report file is NOT written)
  3. split            - recomputed seed-42 split and the saved split file match the fingerprint
  4. historical       - every frozen Phase 3/4 result matches reports/historical_results.sha256
  5. artifact         - hash, scikit-learn version and feature order (ChurnPredictor.load),
                        threshold == Phase 4 decision, trained on the fingerprinted training IDs
  6. coefficients     - artifact coefficients equal Phase 4's logreg_coefficients.csv
  7. api              - in-process requests: /health, /model-info, one valid and one invalid /predict

What it deliberately does NOT do: fit models, compute evaluation metrics, or
score any dataset rows (test or training). The only predictions are the
single made-up customer sent to the API. It writes no files.

Passing proves the pieces are intact and wired together. It does NOT prove
the model is useful in production.
"""

import sys

sys.dont_write_bytecode = True   # keep later imports from writing .pyc files into the project

import hashlib  # noqa: E402
import json  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churn_platform.config import (  # noqa: E402
    ARTIFACT_FILE,
    HISTORICAL_HASHES_FILE,
    PROJECT_ROOT,
    RAW_DATA_FILE,
    REPORTS_DIR,
    SPLIT_FILE,
    SPLIT_FINGERPRINT_FILE,
)


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw: Path = RAW_DATA_FILE
    split: Path = SPLIT_FILE
    fingerprint: Path = SPLIT_FINGERPRINT_FILE
    historical: Path = HISTORICAL_HASHES_FILE
    artifact: Path = ARTIFACT_FILE
    decisions: Path = REPORTS_DIR / "phase4" / "decisions.json"
    coefficients: Path = REPORTS_DIR / "phase4" / "logreg_coefficients.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Context:
    """Lazily loads shared inputs once; checks that need them fail cleanly if they can't load."""

    def __init__(self, paths: Paths):
        self.paths = paths
        self._fingerprint = self._raw = None

    @property
    def fingerprint(self) -> dict:
        if self._fingerprint is None:
            self._fingerprint = json.loads(self.paths.fingerprint.read_text(encoding="utf-8"))
        return self._fingerprint

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            from churn_platform.validation import load_raw
            self._raw = load_raw(self.paths.raw)
        return self._raw


# --- individual checks: each returns a short detail string or raises -----------------

def check_raw(ctx: _Context) -> str:
    actual = _sha256(ctx.paths.raw)
    assert actual == ctx.fingerprint["raw_file_sha256"], f"raw SHA-256 is {actual[:16]}..., expected a different value"
    return f"sha256 {actual[:8]}...{actual[-4:]}"


def check_validation(ctx: _Context) -> str:
    from churn_platform.validation import validate
    issues = validate(ctx.raw)
    errors = [i for i in issues if i.severity == "ERROR"]
    assert not errors, f"{len(errors)} validation error(s): {errors[0].check}"
    return f"0 errors, {len(issues)} known warnings"


def check_split(ctx: _Context) -> str:
    from churn_platform.make_split import compute_split, split_matches
    assert split_matches(compute_split(ctx.raw), ctx.fingerprint), "recomputed split differs from fingerprint"
    assert ctx.paths.split.is_file(), "split file missing; run python -m churn_platform.make_split"
    assert split_matches(pd.read_csv(ctx.paths.split), ctx.fingerprint), "saved split file differs from fingerprint"
    return f"{ctx.fingerprint['train_rows']} train / {ctx.fingerprint['test_rows']} test"


def check_historical(ctx: _Context) -> str:
    expected = [line.split(maxsplit=1) for line in ctx.paths.historical.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
    bad = [rel for digest, rel in expected
           if not (ctx.paths.root / rel).is_file() or _sha256(ctx.paths.root / rel) != digest]
    assert expected, "manifest is empty"
    assert not bad, f"{len(bad)} changed or missing, e.g. {bad[0]}"
    return f"{len(expected)} frozen files unchanged"


def check_artifact(ctx: _Context) -> str:
    from churn_platform.api.predictor import ChurnPredictor
    predictor = ChurnPredictor.load(ctx.paths.artifact)   # verifies hash, sklearn version, feature order
    decision = json.loads(ctx.paths.decisions.read_text(encoding="utf-8"))["selected_threshold"]
    assert predictor.threshold == decision, f"artifact threshold {predictor.threshold} != Phase 4 decision {decision}"
    trained_on = predictor.metadata["training_data"]["training_ids_sha256"]
    assert trained_on == ctx.fingerprint["train_ids_sha256"], "artifact was not trained on the fingerprinted training rows"
    assert "customerID" not in predictor.metadata["feature_order"]
    ctx.predictor = predictor
    return f"v{predictor.metadata['model_version']}, threshold {predictor.threshold}, training IDs match"


def check_coefficients(ctx: _Context) -> str:
    from churn_platform.features import build_model_frame
    from churn_platform.interpretation import logreg_coefficients
    predictor = getattr(ctx, "predictor", None)
    assert predictor is not None, "artifact check failed, nothing to compare"
    X, _, ids = build_model_frame(ctx.raw)
    split = pd.read_csv(ctx.paths.split)
    X_train = X[ids.isin(set(split.loc[split["split"] == "train", "customerID"]))]
    # X_train is only used to pick each feature's most common category as the reference; nothing is predicted.
    ours = logreg_coefficients(predictor.pipeline, X_train).set_index(["feature", "term"])["log_odds"].sort_index()
    theirs = pd.read_csv(ctx.paths.coefficients).set_index(["feature", "term"])["log_odds"].sort_index()
    assert ours.index.equals(theirs.index), "different coefficient terms"
    worst = float(np.max(np.abs(ours.to_numpy() - theirs.to_numpy())))
    assert worst <= 1e-9, f"max coefficient difference {worst:.2e}"
    return f"{len(ours)} terms, max |diff| {worst:.1e}"


def check_api(ctx: _Context) -> str:
    from fastapi.testclient import TestClient

    from churn_platform.api.main import create_app
    from churn_platform.api.schemas import CustomerFeatures
    example = CustomerFeatures.model_config["json_schema_extra"]["example"]   # made-up customer
    with TestClient(create_app(ctx.paths.artifact)) as client:                # in-process, no network
        health = client.get("/health")
        info = client.get("/model-info")
        ok = client.post("/predict", json=example)
        bad = client.post("/predict", json={**example, "Contract": "Three year"})
    assert health.status_code == 200 and health.json()["model_loaded"], f"/health {health.status_code}"
    assert info.status_code == 200 and info.json()["threshold"] == ctx.predictor.threshold, f"/model-info {info.status_code}"
    p = ok.json().get("churn_probability", -1)
    assert ok.status_code == 200 and 0.0 <= p <= 1.0, f"/predict valid -> {ok.status_code}"
    assert bad.status_code == 422, f"/predict invalid -> {bad.status_code}, expected 422"
    return "health 200, model-info 200, predict 200, invalid 422"


CHECKS = [
    ("raw data", check_raw),
    ("validation", check_validation),
    ("split", check_split),
    ("historical results", check_historical),
    ("model artifact", check_artifact),
    ("coefficient parity", check_coefficients),
    ("API smoke test", check_api),
]


def run_checks(paths: Paths = Paths()) -> list[dict]:
    ctx = _Context(paths)
    results = []
    for name, check in CHECKS:
        try:
            results.append({"check": name, "passed": True, "detail": check(ctx)})
        except Exception as exc:   # report and continue: one failure shouldn't hide the others
            results.append({"check": name, "passed": False, "detail": str(exc) or type(exc).__name__})
    return results


def main() -> int:
    import logging
    logging.disable(logging.WARNING)   # keep the table readable; failures are reported in it
    results = run_checks()
    width = max(len(r["check"]) for r in results)
    print(f"{'Check'.ljust(width)}  Result  Detail")
    print(f"{'-' * width}  ------  ------")
    for r in results:
        print(f"{r['check'].ljust(width)}  {'PASS' if r['passed'] else 'FAIL':6}  {r['detail']}")
    failed = sum(not r["passed"] for r in results)
    print(f"\nRESULT: {'PASSED' if not failed else f'FAILED ({failed} of {len(results)} checks)'}")
    print("Note: verifies integrity and wiring only; it does not show the model is useful in production.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
