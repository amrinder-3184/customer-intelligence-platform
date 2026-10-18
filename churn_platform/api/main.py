"""FastAPI application: /health, /model-info, /predict.

Run locally (from the project root, after `python -m churn_platform.artifact`):
    uvicorn churn_platform.api.main:app --host 127.0.0.1 --port 8000
Interactive docs: http://127.0.0.1:8000/docs

Design:
- create_app() builds the app; tests call it with their own artifact path.
- The model is loaded ONCE at startup (lifespan) and kept on app.state:
  no retraining, no refitting per request. If loading fails, the service
  still starts but reports itself unavailable (503) instead of crashing or
  pretending to be ready.
- Errors returned to callers never include tracebacks or file paths; details
  go to the server log only.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn_platform.api.predictor import ChurnPredictor, ModelUnavailableError, PredictionError
from churn_platform.api.schemas import (
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from churn_platform.config import ARTIFACT_FILE

logger = logging.getLogger("churn_api")


class ServiceUnavailable(Exception):
    pass


def create_app(artifact_path: Path | None = None) -> FastAPI:
    path = Path(artifact_path or os.environ.get("CHURN_MODEL_PATH", ARTIFACT_FILE))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.predictor = ChurnPredictor.load(path)
            logger.info("model loaded: %s", app.state.predictor.metadata["model_version"])
        except Exception:  # any failure -> unavailable, never a half-loaded model
            logger.exception("model failed to load")
            app.state.predictor = None
        yield

    app = FastAPI(title="Customer churn prediction API", version="1.0.0", lifespan=lifespan,
                  description="Serves a retrospective logistic-regression churn model. "
                              "Predictions are model estimates, not guarantees.")

    def get_predictor(request: Request) -> ChurnPredictor:
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise ServiceUnavailable()
        return predictor

    # --- error handlers: clean JSON, no internals ------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
                   "message": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": errors})

    @app.exception_handler(ServiceUnavailable)
    async def unavailable(_: Request, __: ServiceUnavailable):
        return JSONResponse(status_code=503, content={"detail": "Model is not available"})

    @app.exception_handler(PredictionError)
    async def prediction_failed(_: Request, exc: PredictionError):
        logger.error("prediction failed: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Prediction failed"})

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # --- routes ------------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            return JSONResponse(status_code=503, content=HealthResponse(
                status="unavailable", model_loaded=False).model_dump())
        return HealthResponse(status="ok", model_loaded=True, model_version=predictor.metadata["model_version"])

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(predictor: ChurnPredictor = Depends(get_predictor)):
        m = predictor.metadata
        t = m["training_data"]
        return ModelInfoResponse(
            model_name=m["model_name"], model_version=m["model_version"], created_at_utc=m["created_at_utc"],
            n_predictors=len(m["feature_order"]),
            # Only schema fields; no file names, hashes of private data, or paths.
            predictors=[{k: f[k] for k in ("name", "type", "allowed", "min", "max", "nullable") if k in f}
                        for f in m["features"]],
            target=m["target"], threshold=predictor.threshold,
            threshold_method=m["decision"]["threshold_method"],
            cost_assumptions=m["decision"]["cost_assumptions"], risk_bands=m["decision"]["risk_bands"],
            training_summary={"rows": t["rows"], "churn_rate": round(t["churn_rate"], 4),
                              "partition": t["partition"], "test_rows_excluded": t["excluded_test_rows"]},
            historical_evaluation=m["historical_evaluation"], limitations=m["limitations"],
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(customer: CustomerFeatures, predictor: ChurnPredictor = Depends(get_predictor)):
        return PredictionResponse(**predictor.predict(customer.model_dump()))

    return app


app = create_app()
