"""Phase 5: fit the selected model on the TRAINING partition and save it for serving.

Usage (from the project root):
    python -m churn_platform.artifact

Writes two files to models/ (git-ignored, reproducible with this command):
    churn_logreg_v1.joblib          the FITTED scikit-learn Pipeline (preprocessing + model)
    churn_logreg_v1.metadata.json   human-readable metadata: schema, threshold, provenance

Why save preprocessing and model together: the imputer's median, the
scaler's means/SDs and the one-hot category lists were LEARNED from the
training data. Inference must reuse exactly those learned values. Refitting
them on request data would give different numbers for the same customer.

What this script does NOT do: it does not choose a model or threshold
(both come from Phase 4's decisions.json), it does not evaluate anything on
the held-out test set, and it never reads test-row features or labels for
fitting. Test customerIDs are read only to verify the split.

Security: joblib/pickle files can execute code when loaded. Treat the
.joblib file as a TRUSTED LOCAL file produced by this script. Never load an
artifact uploaded by a user. The SHA-256 stored in the metadata detects
accidental corruption or replacement; it is not protection against someone
who can rewrite both files.
"""

import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning

from churn_platform.analysis import CATEGORICAL_FEATURES
from churn_platform.config import ARTIFACT_FILE, RANDOM_SEED, RAW_DATA_FILE, REPORTS_DIR, SPLIT_FILE
from churn_platform.download import sha256_of
from churn_platform.features import MODEL_FEATURES, NUMERIC_FEATURES, assert_no_test_rows, build_model_frame, split_data
from churn_platform.interpretation import logreg_coefficients
from churn_platform.modeling import build_logreg
from churn_platform.validation import CATEGORICAL_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS, load_raw, validate

MODEL_NAME = "churn-logreg"
MODEL_VERSION = "1.0.0"
METADATA_FORMAT_VERSION = 1
DECISIONS_FILE = REPORTS_DIR / "phase4" / "decisions.json"
PHASE4_COEFFICIENTS = REPORTS_DIR / "phase4" / "logreg_coefficients.csv"

LIMITATIONS = [
    "Retrospective model: for customers who churned, tenure and TotalCharges were recorded at churn time, "
    "so this is not a validated early-warning model.",
    "The 0.16 threshold minimises cost under an ILLUSTRATIVE 5:1 false-negative:false-positive cost "
    "assumption, not measured business costs.",
    "Trained on a single synthetic IBM snapshot (7,043 customers). Real-world performance is unknown.",
    "Scores are associations learned from data, not causal effects; changing a feature for a customer is "
    "not guaranteed to change their churn.",
]


class TrainingPartitionError(RuntimeError):
    """Raised when the training rows cannot be reconstructed safely."""


def metadata_path_for(artifact_path: Path) -> Path:
    return artifact_path.with_name(artifact_path.stem + ".metadata.json")


def load_training_partition(raw: pd.DataFrame, split_file: Path = SPLIT_FILE):
    """Return (X_train, y_train, id_train, info), verified against the saved Phase 3 split.

    Two independent sources must agree: the seed-42 split recomputed now, and
    the customerIDs saved in data/processed/train_test_split.csv. If they
    don't, we stop rather than guess.
    """
    X, y, ids = build_model_frame(raw)
    X_train, X_test, y_train, _, id_train, id_test = split_data(X, y, ids)

    if not split_file.exists():
        raise TrainingPartitionError(
            f"{split_file.name} not found, so the training partition cannot be verified. "
            "It is produced by Phase 3 (python -m churn_platform.train_baseline).")
    saved = pd.read_csv(split_file)
    saved_train = set(saved.loc[saved["split"] == "train", ID_COLUMN])
    saved_test = set(saved.loc[saved["split"] == "test", ID_COLUMN])
    if saved_train != set(id_train) or saved_test != set(id_test):
        raise TrainingPartitionError("Recomputed split does not match the saved Phase 3 split; refusing to fit.")
    assert_no_test_rows(X_train, X_test.index)

    info = {
        "rows": int(len(X_train)),
        "churn_rate": float(y_train.mean()),
        "excluded_test_rows": int(len(X_test)),
        "split_file_sha256": sha256_of(split_file),
        # Hash of the sorted training IDs: lets anyone check which rows were used without listing them.
        "training_ids_sha256": hashlib.sha256("\n".join(sorted(id_train)).encode("utf-8")).hexdigest(),
    }
    return X_train, y_train, id_train, info


def fit_serving_pipeline(X_train: pd.DataFrame, y_train: pd.Series):
    """The exact Phase 3/4 model: build_logreg() with median+indicator imputation, unweighted, C=1."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return build_logreg().fit(X_train, y_train)


def feature_schema(pipeline) -> list[dict]:
    """Ordered predictor schema, taken from the Phase 1 rules and the fitted encoder."""
    encoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    learned = {f: [str(c) for c in cats] for f, cats in zip(encoder.feature_names_in_, encoder.categories_)}
    schema = []
    for name in MODEL_FEATURES:
        if name in NUMERIC_FEATURES:
            low, high = NUMERIC_COLUMNS[name]
            schema.append({"name": name, "type": "integer" if name == "tenure" else "number",
                           "min": low, "max": high, "nullable": name == "TotalCharges"})
        else:
            schema.append({"name": name, "type": "category", "allowed": sorted(CATEGORICAL_COLUMNS[name]),
                           "learned_by_encoder": learned[name], "nullable": False})
    return schema


def compare_with_phase4(pipeline, X_train) -> bool | None:
    """True if the coefficients equal Phase 4's (same model, same rows). None if Phase 4 output is absent."""
    if not PHASE4_COEFFICIENTS.exists():
        return None
    ours = logreg_coefficients(pipeline, X_train).set_index(["feature", "term"])["log_odds"]
    theirs = pd.read_csv(PHASE4_COEFFICIENTS).set_index(["feature", "term"])["log_odds"]
    if set(ours.index) != set(theirs.index):
        return False
    return bool(np.allclose(ours.sort_index(), theirs.sort_index(), rtol=0, atol=1e-9))


def build_metadata(pipeline, training: dict, decisions: dict, raw_sha: str) -> dict:
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    num = pre.named_transformers_["num"]
    threshold = float(decisions["selected_threshold"])
    return {
        "metadata_format_version": METADATA_FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": "Churn == 'Yes' (1) vs 'No' (0); model outputs P(churn)",
        "features": feature_schema(pipeline),
        "feature_order": MODEL_FEATURES,
        "excluded_columns": {ID_COLUMN: "identifier, never a predictor", "Churn": "target"},
        "preprocessing": {
            "numeric": "median imputation + missing indicator, then StandardScaler (fitted on training rows)",
            "categorical": "OneHotEncoder(handle_unknown='ignore'), all categories kept",
            "train_median_TotalCharges": float(num.named_steps["impute"].statistics_[NUMERIC_FEATURES.index("TotalCharges")]),
            "n_model_inputs": int(len(pre.get_feature_names_out())),
        },
        "model": {"class": type(model).__name__,
                  "params": {k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, type(None)))},
                  "solver_iterations": int(model.n_iter_[0])},
        "decision": {
            "threshold": threshold,
            "threshold_source": "reports/phase4/decisions.json",
            "threshold_method": decisions["threshold_rule"] + " (training out-of-fold probabilities)",
            "decisions_sha256": sha256_of(DECISIONS_FILE),
            "cost_assumptions": decisions["cost_assumptions"],
            "risk_bands": {"low": f"p < {threshold}", "elevated": f"{threshold} <= p < 0.5", "high": "p >= 0.5"},
            "risk_band_note": "Only the threshold decides predicted_churn; the 0.5 band edge is descriptive.",
        },
        "training_data": {"raw_file_sha256": raw_sha, "partition": "Phase 3 training split (seed 42)",
                          "seed": RANDOM_SEED, **training},
        "verification": {"coefficients_match_phase4": None},
        "historical_evaluation": historical_evaluation(),
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "artifact": {"format": "joblib-pickled scikit-learn Pipeline; trusted local file only",
                     "file": None, "sha256": None},
        "limitations": LIMITATIONS,
    }


def historical_evaluation() -> dict:
    """Copy (read-only) the already-reported held-out results. Nothing is re-evaluated."""
    out = {"note": "Copied from earlier reports; the test set is NOT re-scored when building this artifact."}
    p4 = REPORTS_DIR / "phase4" / "final_test_evaluation.json"
    if p4.exists():
        s = json.loads(p4.read_text(encoding="utf-8"))["at_selected_threshold"]
        out["phase4_test_at_selected_threshold"] = {k: s[k] for k in ("threshold", "roc_auc", "average_precision",
                                                                       "precision", "recall", "confusion_matrix")}
    p3 = REPORTS_DIR / "phase3" / "metrics.json"
    if p3.exists():
        s = json.loads(p3.read_text(encoding="utf-8"))["test_metrics"]["logreg"]
        out["phase3_test_at_0_5"] = {k: s[k] for k in ("threshold", "roc_auc", "precision", "recall", "confusion_matrix")}
    return out


def save_artifact(pipeline, metadata: dict, artifact_path: Path) -> dict:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    metadata["artifact"]["file"] = artifact_path.name           # file name only, never a full path
    metadata["artifact"]["sha256"] = sha256_of(artifact_path)
    metadata_path_for(artifact_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_artifact(artifact_path: Path = ARTIFACT_FILE) -> dict:
    raw_sha = sha256_of(RAW_DATA_FILE)
    raw = load_raw(RAW_DATA_FILE)
    if any(i.severity == "ERROR" for i in validate(raw)):
        raise RuntimeError("Raw data failed validation; run python -m churn_platform.inspect_data")
    decisions = json.loads(DECISIONS_FILE.read_text(encoding="utf-8"))

    X_train, y_train, _, training = load_training_partition(raw)
    pipeline = fit_serving_pipeline(X_train, y_train)
    metadata = build_metadata(pipeline, training, decisions, raw_sha)
    metadata["verification"]["coefficients_match_phase4"] = compare_with_phase4(pipeline, X_train)
    return save_artifact(pipeline, metadata, artifact_path)


def main() -> int:
    try:
        meta = build_artifact()
    except TrainingPartitionError as exc:
        print(f"STOPPED: {exc}")
        return 1
    t, d = meta["training_data"], meta["decision"]
    print(f"Artifact:  {ARTIFACT_FILE.relative_to(ARTIFACT_FILE.parents[1])} (sha256 {meta['artifact']['sha256'][:16]}...)")
    print(f"Metadata:  {metadata_path_for(ARTIFACT_FILE).name}")
    print(f"Trained on {t['rows']} training rows (churn rate {t['churn_rate']:.4f}); "
          f"{t['excluded_test_rows']} test rows excluded")
    print(f"Features:  {len(meta['feature_order'])} predictors -> {meta['preprocessing']['n_model_inputs']} model inputs")
    print(f"Threshold: {d['threshold']} ({d['cost_assumptions']['status']})")
    print(f"Coefficients match Phase 4: {meta['verification']['coefficients_match_phase4']}")
    ok = meta["verification"]["coefficients_match_phase4"] is not False
    print("RESULT:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
