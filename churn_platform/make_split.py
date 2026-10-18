"""Phase 6: recreate the Phase 3 train/test split file WITHOUT fitting or scoring anything.

Usage (from the project root):
    python -m churn_platform.make_split

Why this exists: data/processed/ is git-ignored, and the only other code that
writes train_test_split.csv is Phase 3's train_baseline.py, which also scores
the held-out test set. A fresh clone needs the split (artifact.py refuses to
fit without it), so this command rebuilds it safely.

Steps, each of which must pass before the next:
  1. Raw CSV SHA-256 equals the value in reports/split_fingerprint.json.
  2. The Phase 1 validator reports no errors.
  3. The split is recomputed with the existing split_data() (seed 42, stratified 80/20).
     Stratification reads the churn labels only to balance the two parts,
     exactly as in Phase 3; no model is fitted and no row is scored.
  4. The train and test customerID sets must match the committed fingerprint.
  5. If the split file already exists it is compared, never overwritten.
     If it is missing, it is written in the same format train_baseline.py uses.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from churn_platform.config import RAW_DATA_FILE, SPLIT_FILE, SPLIT_FINGERPRINT_FILE
from churn_platform.download import sha256_of
from churn_platform.features import build_model_frame, split_data
from churn_platform.validation import ID_COLUMN, load_raw, validate


class SplitError(RuntimeError):
    """The split cannot be (re)created safely. Nothing has been written."""


def ids_sha256(ids) -> str:
    """Order-independent fingerprint of a set of customerIDs."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def load_fingerprint(path: Path = SPLIT_FINGERPRINT_FILE) -> dict:
    if not path.is_file():
        raise SplitError(f"{path.name} not found; it is committed with the repository")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_split(raw: pd.DataFrame) -> pd.DataFrame:
    """The Phase 3 split as a (customerID, split) table: training rows first, then test rows."""
    X, y, ids = build_model_frame(raw)
    _, _, _, _, id_train, id_test = split_data(X, y, ids)
    return pd.concat([
        pd.DataFrame({"customerID": id_train, "split": "train"}),
        pd.DataFrame({"customerID": id_test, "split": "test"}),
    ])


def split_matches(split: pd.DataFrame, fingerprint: dict) -> bool:
    train = split.loc[split["split"] == "train", ID_COLUMN]
    test = split.loc[split["split"] == "test", ID_COLUMN]
    return (len(train) == fingerprint["train_rows"] and len(test) == fingerprint["test_rows"]
            and ids_sha256(train) == fingerprint["train_ids_sha256"]
            and ids_sha256(test) == fingerprint["test_ids_sha256"])


def make_split(raw_file: Path = RAW_DATA_FILE, split_file: Path = SPLIT_FILE,
               fingerprint_file: Path = SPLIT_FINGERPRINT_FILE) -> str:
    """Return "written" or "already present"; raise SplitError (writing nothing) otherwise."""
    fingerprint = load_fingerprint(fingerprint_file)
    if not raw_file.is_file():
        raise SplitError("raw CSV missing; run python -m churn_platform.download")
    if sha256_of(raw_file) != fingerprint["raw_file_sha256"]:
        raise SplitError("raw CSV SHA-256 differs from the fingerprint; refusing to build a split")
    raw = load_raw(raw_file)
    if any(i.severity == "ERROR" for i in validate(raw)):
        raise SplitError("raw data failed validation; run python -m churn_platform.inspect_data")

    split = compute_split(raw)
    if not split_matches(split, fingerprint):
        raise SplitError("recomputed split does not match the committed fingerprint "
                         "(different library versions?); nothing written")

    if split_file.exists():
        if not split_matches(pd.read_csv(split_file), fingerprint):
            raise SplitError(f"existing {split_file.name} does not match the fingerprint; left untouched")
        return "already present"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(split_file, index=False)          # same call and format as train_baseline.py
    return "written"


def main() -> int:
    try:
        status = make_split()
    except SplitError as exc:
        print(f"FAILED: {exc}")
        return 1
    fp = load_fingerprint()
    print(f"Split file {status}: {SPLIT_FILE.name} "
          f"({fp['train_rows']} train / {fp['test_rows']} test, matches fingerprint)")
    print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
