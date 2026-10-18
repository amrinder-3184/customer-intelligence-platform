"""Tests for churn_platform.make_split: reproduces the Phase 3 split without fitting or scoring.

All writes go to pytest temporary directories; the real split file is only read.
"""

import hashlib
import json
import shutil

import pandas as pd
import pytest

import churn_platform.artifact as artifact_module
import churn_platform.config as config
import churn_platform.train_baseline as train_baseline
from churn_platform.make_split import SplitError, ids_sha256, load_fingerprint, make_split

needs_data = pytest.mark.skipif(
    not (config.RAW_DATA_FILE.exists() and config.SPLIT_FILE.exists()),
    reason="needs the raw CSV and the Phase 3 split file")


def test_split_path_defined_consistently():
    # train_baseline.py was deliberately left unchanged; its constant must stay equal to config's.
    assert train_baseline.SPLIT_FILE == config.SPLIT_FILE == artifact_module.SPLIT_FILE


def test_ids_hash_is_order_independent():
    assert ids_sha256(["b", "a", "c"]) == ids_sha256(["c", "b", "a"])
    assert ids_sha256(["a", "b"]) != ids_sha256(["a", "c"])


@needs_data
def test_fingerprint_describes_the_saved_phase3_split():
    fp = load_fingerprint()
    saved = pd.read_csv(config.SPLIT_FILE)
    assert ids_sha256(saved.loc[saved["split"] == "train", "customerID"]) == fp["train_ids_sha256"]
    assert ids_sha256(saved.loc[saved["split"] == "test", "customerID"]) == fp["test_ids_sha256"]
    assert (fp["train_rows"], fp["test_rows"]) == (5634, 1409)
    assert fp["raw_file_sha256"].startswith("16320c9c")


@needs_data
def test_fresh_write_is_byte_identical_to_phase3_file(tmp_path):
    target = tmp_path / "processed" / "train_test_split.csv"
    assert make_split(split_file=target) == "written"
    assert target.read_bytes() == config.SPLIT_FILE.read_bytes()


@needs_data
def test_existing_matching_file_is_left_alone(tmp_path):
    target = tmp_path / "train_test_split.csv"
    shutil.copy(config.SPLIT_FILE, target)
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    assert make_split(split_file=target) == "already present"
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


@needs_data
def test_mismatching_existing_file_is_refused_and_untouched(tmp_path):
    target = tmp_path / "train_test_split.csv"
    split = pd.read_csv(config.SPLIT_FILE)
    split.loc[split.index[split["split"] == "test"][0], "split"] = "train"   # move one customer
    split.to_csv(target, index=False)
    before = target.read_bytes()
    with pytest.raises(SplitError, match="does not match"):
        make_split(split_file=target)
    assert target.read_bytes() == before


@needs_data
def test_wrong_raw_checksum_is_refused_and_nothing_written(tmp_path):
    raw = tmp_path / "raw.csv"
    raw.write_bytes(config.RAW_DATA_FILE.read_bytes() + b"\n")
    target = tmp_path / "split.csv"
    with pytest.raises(SplitError, match="SHA-256"):
        make_split(raw_file=raw, split_file=target)
    assert not target.exists()


@needs_data
def test_invalid_raw_data_is_refused_before_writing(tmp_path):
    raw_text = config.RAW_DATA_FILE.read_text(encoding="utf-8").replace("Month-to-month", "Three year", 1)
    raw = tmp_path / "raw.csv"
    raw.write_text(raw_text, encoding="utf-8", newline="")
    fp = {**load_fingerprint(), "raw_file_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
    fp_file = tmp_path / "fingerprint.json"
    fp_file.write_text(json.dumps(fp), encoding="utf-8")
    target = tmp_path / "split.csv"
    with pytest.raises(SplitError, match="validation"):
        make_split(raw_file=raw, split_file=target, fingerprint_file=fp_file)
    assert not target.exists()


@needs_data
def test_make_split_never_fits_or_predicts(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    def forbidden(*args, **kwargs):
        raise AssertionError("make_split must not fit or score anything")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(Pipeline, "predict_proba", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    assert make_split(split_file=tmp_path / "split.csv") == "written"
