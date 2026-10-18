"""Tests for churn_platform.artifact: training-only fitting, metadata, loading.

Building the real artifact needs the dataset, the saved Phase 3 split and
Phase 4's decisions.json; tests that need them are skipped otherwise.
Artifacts are written to pytest's temporary directory, never to models/.
"""

import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from churn_platform import artifact
from churn_platform.api.predictor import ChurnPredictor, ModelUnavailableError
from churn_platform.config import PROJECT_ROOT, RAW_DATA_FILE
from churn_platform.features import MODEL_FEATURES, build_model_frame
from churn_platform.validation import CATEGORICAL_COLUMNS, load_raw

needs_inputs = pytest.mark.skipif(
    not (RAW_DATA_FILE.exists() and artifact.SPLIT_FILE.exists() and artifact.DECISIONS_FILE.exists()),
    reason="needs raw data, Phase 3 split file and Phase 4 decisions.json")


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "churn_logreg_v1.joblib"
    metadata = artifact.build_artifact(path)
    return path, metadata


@pytest.fixture(scope="session")
def training_rows():
    saved = pd.read_csv(artifact.SPLIT_FILE)
    X, y, ids = build_model_frame(load_raw(RAW_DATA_FILE))
    train_ids = set(saved.loc[saved["split"] == "train", "customerID"])
    mask = ids.isin(train_ids)
    return X[mask], y[mask], ids[mask], X[~mask]


# --- 1 & 16: fitted on training rows only -----------------------------------------

@needs_inputs
def test_artifact_trained_on_exact_training_partition(built, training_rows):
    _, meta = built
    _, _, train_ids, X_test = training_rows
    t = meta["training_data"]
    assert t["rows"] == len(train_ids) == 5634
    assert t["excluded_test_rows"] == len(X_test) == 1409
    expected = hashlib.sha256("\n".join(sorted(train_ids)).encode("utf-8")).hexdigest()
    assert t["training_ids_sha256"] == expected


@needs_inputs
def test_learned_statistics_come_from_training_rows_not_all_rows(built, training_rows):
    path, _ = built
    X_train, _, _, X_test = training_rows
    pipeline = ChurnPredictor.load(path).pipeline
    num = pipeline.named_steps["preprocess"].named_transformers_["num"]
    assert num.named_steps["impute"].statistics_[2] == pytest.approx(X_train["TotalCharges"].median())
    assert num.named_steps["scale"].mean_[1] == pytest.approx(X_train["MonthlyCharges"].mean())
    all_rows_mean = pd.concat([X_train, X_test])["MonthlyCharges"].mean()
    assert num.named_steps["scale"].mean_[1] != pytest.approx(all_rows_mean, abs=1e-9)


@needs_inputs
def test_refuses_when_saved_split_disagrees(tmp_path):
    tampered = tmp_path / "split.csv"
    split = pd.read_csv(artifact.SPLIT_FILE)
    first_test = split.index[split["split"] == "test"][0]
    split.loc[first_test, "split"] = "train"           # move one test customer into training
    split.to_csv(tampered, index=False)
    with pytest.raises(artifact.TrainingPartitionError):
        artifact.load_training_partition(load_raw(RAW_DATA_FILE), tampered)


@needs_inputs
def test_refuses_when_split_file_missing(tmp_path):
    with pytest.raises(artifact.TrainingPartitionError, match="not found"):
        artifact.load_training_partition(load_raw(RAW_DATA_FILE), tmp_path / "missing.csv")


@needs_inputs
def test_same_model_as_phase4(built):
    _, meta = built
    assert meta["verification"]["coefficients_match_phase4"] is True


# --- 2 & 15: metadata and feature schema -------------------------------------------

@needs_inputs
def test_metadata_schema_and_decision(built):
    _, meta = built
    decisions = json.loads(artifact.DECISIONS_FILE.read_text(encoding="utf-8"))
    assert meta["feature_order"] == MODEL_FEATURES and len(MODEL_FEATURES) == 19
    assert "customerID" not in meta["feature_order"]
    assert meta["decision"]["threshold"] == decisions["selected_threshold"] == 0.16
    assert meta["decision"]["cost_assumptions"] == {"fp_cost": 1.0, "fn_cost": 5.0,
                                                    "status": "ILLUSTRATIVE ASSUMPTION, not measured business costs"}
    assert meta["model"]["class"] == "LogisticRegression" and meta["model"]["params"]["class_weight"] is None
    assert meta["training_data"]["raw_file_sha256"].startswith("16320c9c")
    assert {"python", "scikit-learn", "pandas", "numpy", "joblib"} <= set(meta["environment"])


@needs_inputs
def test_schema_categories_match_validator_and_encoder(built):
    _, meta = built
    for f in meta["features"]:
        if f["type"] == "category":
            assert set(f["allowed"]) == CATEGORICAL_COLUMNS[f["name"]]
            assert set(f["learned_by_encoder"]) == CATEGORICAL_COLUMNS[f["name"]]


@needs_inputs
def test_metadata_contains_no_filesystem_paths(built):
    path, meta = built
    text = json.dumps(meta)
    assert str(PROJECT_ROOT) not in text and str(path.parent) not in text
    assert meta["artifact"]["file"] == path.name


@needs_inputs
def test_rebuilding_gives_identical_model(built, tmp_path):
    first = ChurnPredictor.load(built[0]).pipeline.named_steps["model"].coef_
    other = tmp_path / "again.joblib"
    artifact.build_artifact(other)
    np.testing.assert_array_equal(first, ChurnPredictor.load(other).pipeline.named_steps["model"].coef_)


# --- 3: loading ----------------------------------------------------------------------

@needs_inputs
def test_load_in_a_fresh_process(built):
    path, _ = built
    code = ("from churn_platform.api.predictor import ChurnPredictor; import sys; "
            f"p = ChurnPredictor.load(r'{path}'); print(p.threshold, len(p.metadata['feature_order']))")
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["0.16", "19"]


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(ModelUnavailableError):
        ChurnPredictor.load(tmp_path / "nope.joblib")


@needs_inputs
def test_corrupted_artifact_is_not_unpickled(built, tmp_path):
    path, _ = built
    copy = tmp_path / path.name
    shutil.copy(path, copy)
    shutil.copy(artifact.metadata_path_for(path), artifact.metadata_path_for(copy))
    with open(copy, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ModelUnavailableError, match="SHA-256"):
        ChurnPredictor.load(copy)


@needs_inputs
def test_version_mismatch_is_refused(built, tmp_path):
    path, meta = built
    copy = tmp_path / path.name
    shutil.copy(path, copy)
    changed = json.loads(json.dumps(meta))
    changed["environment"]["scikit-learn"] = "0.0.1"
    artifact.metadata_path_for(copy).write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ModelUnavailableError, match="scikit-learn"):
        ChurnPredictor.load(copy)
