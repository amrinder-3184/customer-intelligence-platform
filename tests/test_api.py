"""Tests for the prediction service.

Two layers are tested separately:
- TestPredictor: model logic only (no HTTP), e.g. threshold edges, row conversion.
- HTTP tests via FastAPI's TestClient: status codes, validation, error bodies.
No network access is needed; TestClient calls the app in-process.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn_platform import artifact
from churn_platform.api.main import create_app
from churn_platform.api.predictor import ChurnPredictor, PredictionError
from churn_platform.api.schemas import CustomerFeatures
from churn_platform.config import RAW_DATA_FILE
from churn_platform.features import MODEL_FEATURES, build_model_frame
from churn_platform.validation import CATEGORICAL_COLUMNS, load_raw

pytestmark = pytest.mark.skipif(
    not (RAW_DATA_FILE.exists() and artifact.SPLIT_FILE.exists() and artifact.DECISIONS_FILE.exists()),
    reason="needs raw data, Phase 3 split file and Phase 4 decisions.json")

VALID = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 430.25,
}


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("api_model") / "churn_logreg_v1.joblib"
    artifact.build_artifact(path)
    return path


@pytest.fixture(scope="module")
def client(artifact_path):
    with TestClient(create_app(artifact_path)) as c:   # "with" runs the startup (lifespan) code
        yield c


@pytest.fixture(scope="module")
def predictor(artifact_path):
    return ChurnPredictor.load(artifact_path)


def post(client, body):
    return client.post("/predict", json=body)


# --- 4, 10, 11: valid requests ---------------------------------------------------------

def test_valid_prediction(client):
    r = post(client, VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability", "predicted_churn", "risk_label", "threshold",
                         "model_name", "model_version", "note"}
    assert 0.0 <= body["churn_probability"] <= 1.0 and math.isfinite(body["churn_probability"])
    assert body["threshold"] == 0.16
    assert body["predicted_churn"] == (body["churn_probability"] >= 0.16)
    assert "not a guarantee" in body["note"]


def test_repeated_requests_are_identical(client):
    assert len({json.dumps(post(client, VALID).json(), sort_keys=True) for _ in range(5)}) == 1


def test_api_path_matches_training_path_on_training_rows(predictor):
    """Same customer via the API conversion and via the training conversion -> same probability."""
    saved = pd.read_csv(artifact.SPLIT_FILE)
    X, _, ids = build_model_frame(load_raw(RAW_DATA_FILE))
    train_ids = set(saved.loc[saved["split"] == "train", "customerID"])
    X_train = X[ids.isin(train_ids)]
    sample = pd.concat([X_train[X_train["TotalCharges"].isna()], X_train.head(300)])   # incl. blank rows
    expected = predictor.pipeline.predict_proba(sample)[:, 1]
    for row, p in zip(sample.to_dict(orient="records"), expected):
        request = {**row, "SeniorCitizen": int(row["SeniorCitizen"]),
                   "TotalCharges": None if pd.isna(row["TotalCharges"]) else row["TotalCharges"]}
        validated = CustomerFeatures(**request).model_dump()
        assert predictor.predict(validated)["churn_probability"] == pytest.approx(p, abs=1e-12)


# --- 8: legitimate missing TotalCharges --------------------------------------------------

def test_null_total_charges_for_new_customer(client, predictor):
    body = {**VALID, "tenure": 0, "TotalCharges": None}
    r = post(client, body)
    assert r.status_code == 200
    # Same as the fitted pipeline on a NaN row: training-median imputation + missing indicator.
    row = predictor.to_model_row(CustomerFeatures(**body).model_dump())
    assert np.isnan(row.loc[0, "TotalCharges"])
    assert r.json()["churn_probability"] == pytest.approx(predictor.pipeline.predict_proba(row)[0, 1])


def test_null_total_charges_rejected_when_tenure_positive(client):
    r = post(client, {**VALID, "TotalCharges": None})
    assert r.status_code == 422
    assert "tenure is 0" in r.json()["errors"][0]["message"]


# --- 5: missing fields ---------------------------------------------------------------------

@pytest.mark.parametrize("field", ["tenure", "Contract", "TotalCharges", "SeniorCitizen"])
def test_missing_field_is_422(client, field):
    body = {k: v for k, v in VALID.items() if k != field}
    r = post(client, body)
    assert r.status_code == 422
    assert {"field": field, "message": "Field required", "type": "missing"} in r.json()["errors"]


# --- 6: categories --------------------------------------------------------------------------

@pytest.mark.parametrize("field, value", [("Contract", "Three year"), ("gender", "female"),
                                          ("PaymentMethod", "Cash"), ("SeniorCitizen", 2),
                                          ("SeniorCitizen", "1"), ("SeniorCitizen", True)])
def test_invalid_category_is_422(client, field, value):
    r = post(client, {**VALID, field: value})
    assert r.status_code == 422
    assert r.json()["errors"][0]["field"] == field


def test_inconsistent_service_combination_is_422(client):
    r = post(client, {**VALID, "InternetService": "No"})   # add-ons still say "No", not "No internet service"
    assert r.status_code == 422


def test_schema_categories_match_training_data():
    schema = CustomerFeatures.model_json_schema()["properties"]
    for name, allowed in CATEGORICAL_COLUMNS.items():
        if name in ("Churn", "SeniorCitizen"):
            continue
        assert set(schema[name]["enum"]) == allowed, name
    assert set(schema["SeniorCitizen"]["enum"]) == {0, 1}


# --- 7: numbers -------------------------------------------------------------------------------

@pytest.mark.parametrize("field, value", [("tenure", -1), ("tenure", 121), ("tenure", "5"), ("tenure", 5.5),
                                          ("MonthlyCharges", -0.01), ("MonthlyCharges", "85"),
                                          ("TotalCharges", 1e9), ("MonthlyCharges", None)])
def test_invalid_number_is_422(client, field, value):
    assert post(client, {**VALID, field: value}).status_code == 422


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
def test_non_finite_numbers_are_422(client, literal):
    raw = json.dumps(VALID).replace('"MonthlyCharges": 85.5', f'"MonthlyCharges": {literal}')
    r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_malformed_json_is_422_without_internals(client):
    r = client.post("/predict", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert "Traceback" not in r.text


# --- 9 & 15: extra fields / customerID ----------------------------------------------------------

@pytest.mark.parametrize("extra", ["customerID", "Churn", "favourite_colour"])
def test_extra_fields_are_rejected(client, extra):
    r = post(client, {**VALID, extra: "x"})
    assert r.status_code == 422
    assert r.json()["errors"][0] == {"field": extra, "message": "Extra inputs are not permitted",
                                     "type": "extra_forbidden"}


def test_customer_id_is_not_a_model_input(predictor):
    assert "customerID" not in list(predictor.pipeline.feature_names_in_)
    assert list(predictor.pipeline.feature_names_in_) == MODEL_FEATURES


# --- 12: threshold logic (no HTTP) ----------------------------------------------------------------

class FixedPipeline:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]])


class TestPredictor:
    @pytest.mark.parametrize("p, churn, label", [(0.16, True, "elevated"), (0.1599999, False, "low"),
                                                 (0.0, False, "low"), (0.49, True, "elevated"),
                                                 (0.5, True, "high"), (1.0, True, "high")])
    def test_threshold_is_inclusive_at_0_16(self, predictor, p, churn, label):
        fake = ChurnPredictor(pipeline=FixedPipeline(p), metadata=predictor.metadata)
        out = fake.predict(CustomerFeatures(**VALID).model_dump())
        assert (out["predicted_churn"], out["risk_label"], out["threshold"]) == (churn, label, 0.16)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -0.1])
    def test_invalid_probability_raises(self, predictor, bad):
        fake = ChurnPredictor(pipeline=FixedPipeline(bad), metadata=predictor.metadata)
        with pytest.raises(PredictionError):
            fake.predict(CustomerFeatures(**VALID).model_dump())

    def test_row_has_training_order_and_types(self, predictor):
        row = predictor.to_model_row(CustomerFeatures(**VALID).model_dump())
        assert list(row.columns) == MODEL_FEATURES
        assert row.loc[0, "SeniorCitizen"] == "0"            # training stored "0"/"1" strings


def test_invalid_model_output_returns_clean_500(artifact_path, predictor):
    with TestClient(create_app(artifact_path), raise_server_exceptions=False) as c:
        c.app.state.predictor = ChurnPredictor(pipeline=FixedPipeline(float("nan")), metadata=predictor.metadata)
        r = post(c, VALID)
    assert r.status_code == 500
    assert r.json() == {"detail": "Prediction failed"}


# --- 13: health ---------------------------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "1.0.0"}


@pytest.mark.parametrize("problem", ["missing", "corrupted"])
def test_service_reports_unavailable_when_model_fails_to_load(tmp_path, artifact_path, problem):
    path = tmp_path / "churn_logreg_v1.joblib"
    if problem == "corrupted":
        path.write_bytes(artifact_path.read_bytes() + b"x")
        artifact.metadata_path_for(path).write_text(
            artifact.metadata_path_for(artifact_path).read_text(encoding="utf-8"), encoding="utf-8")
    with TestClient(create_app(path)) as c:
        health = c.get("/health")
        assert health.status_code == 503
        assert health.json() == {"status": "unavailable", "model_loaded": False, "model_version": None}
        assert post(c, VALID).status_code == 503
        assert c.get("/model-info").status_code == 503
        assert str(tmp_path) not in health.text + post(c, VALID).text


# --- 14: model info -----------------------------------------------------------------------------

def test_model_info(client, artifact_path):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["model_name"] == "churn-logreg" and info["model_version"] == "1.0.0"
    assert info["n_predictors"] == 19 and [p["name"] for p in info["predictors"]] == MODEL_FEATURES
    assert info["threshold"] == 0.16
    assert "ILLUSTRATIVE" in info["cost_assumptions"]["status"]
    assert info["training_summary"]["rows"] == 5634 and info["training_summary"]["test_rows_excluded"] == 1409
    assert any("Retrospective" in s for s in info["limitations"])
    # No filesystem paths, file names or hashes leak to callers.
    for secret in (str(artifact_path.parent), "joblib", "sha256", "training_ids"):
        assert secret not in r.text
