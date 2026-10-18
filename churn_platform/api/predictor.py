"""Model loading and prediction, with no HTTP code.

Keeping this separate from FastAPI means the prediction logic can be tested
directly (tests/test_api.py::TestPredictor) and reused by a batch script
without starting a web server.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn_platform.artifact import METADATA_FORMAT_VERSION, metadata_path_for
from churn_platform.download import sha256_of


class ModelUnavailableError(RuntimeError):
    """The artifact is missing, corrupted, or incompatible. Message is safe to log, not to return."""


class PredictionError(RuntimeError):
    """The model produced an unusable output (e.g. non-finite probability)."""


@dataclass(frozen=True)
class ChurnPredictor:
    pipeline: object
    metadata: dict

    @classmethod
    def load(cls, artifact_path: Path) -> "ChurnPredictor":
        """Load a TRUSTED local artifact after integrity and compatibility checks."""
        artifact_path = Path(artifact_path)
        meta_path = metadata_path_for(artifact_path)
        if not artifact_path.is_file() or not meta_path.is_file():
            raise ModelUnavailableError(f"artifact or metadata missing: {artifact_path.name}")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("metadata_format_version") != METADATA_FORMAT_VERSION:
            raise ModelUnavailableError("unsupported metadata format version")
        # Check the hash BEFORE unpickling, so a corrupted/replaced file is never loaded.
        if sha256_of(artifact_path) != metadata["artifact"]["sha256"]:
            raise ModelUnavailableError("artifact SHA-256 does not match metadata")
        # Pickled scikit-learn objects are only guaranteed to work with the version that created them.
        if metadata["environment"]["scikit-learn"] != sklearn.__version__:
            raise ModelUnavailableError(
                f"artifact built with scikit-learn {metadata['environment']['scikit-learn']}, "
                f"running {sklearn.__version__}; rebuild with python -m churn_platform.artifact")
        pipeline = joblib.load(artifact_path)
        if list(pipeline.feature_names_in_) != metadata["feature_order"]:
            raise ModelUnavailableError("artifact feature order does not match metadata")
        return cls(pipeline=pipeline, metadata=metadata)

    @property
    def threshold(self) -> float:
        return float(self.metadata["decision"]["threshold"])

    def to_model_row(self, features: dict) -> pd.DataFrame:
        """One validated request -> one DataFrame row, in TRAINING column order and types.

        - SeniorCitizen: the API takes 0/1; training saw the strings "0"/"1".
        - TotalCharges: None becomes NaN, which the fitted imputer fills with the
          TRAINING median and flags with the missing indicator, exactly as for
          the 11 blank values during training.
        """
        row = dict(features)
        row["SeniorCitizen"] = str(int(row["SeniorCitizen"]))
        row["tenure"] = int(row["tenure"])
        row["MonthlyCharges"] = float(row["MonthlyCharges"])
        row["TotalCharges"] = np.nan if row["TotalCharges"] is None else float(row["TotalCharges"])
        order = self.metadata["feature_order"]
        missing = set(order) - set(row)
        if missing:
            raise PredictionError(f"missing features: {sorted(missing)}")
        return pd.DataFrame([[row[c] for c in order]], columns=order)

    def risk_label(self, probability: float) -> str:
        if probability < self.threshold:
            return "low"
        return "elevated" if probability < 0.5 else "high"

    def predict(self, features: dict) -> dict:
        """Probability + decision for one customer. Deterministic: no randomness, no refitting."""
        proba = float(self.pipeline.predict_proba(self.to_model_row(features))[0, 1])
        if not math.isfinite(proba) or not 0.0 <= proba <= 1.0:
            raise PredictionError("model returned an invalid probability")
        return {
            "churn_probability": proba,
            "predicted_churn": proba >= self.threshold,
            "risk_label": self.risk_label(proba),
            "threshold": self.threshold,
            "model_name": self.metadata["model_name"],
            "model_version": self.metadata["model_version"],
        }
