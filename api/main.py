"""
Air Quality capstone API.

Exposes two endpoints, matching the serving sketch in the proposal:
  POST /anomaly-check -> flags whether a reading looks anomalous
  POST /forecast       -> predicts PM2.5 24 hours ahead of a reading

Run locally:
    uvicorn api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger docs.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.feature_utils import cyclical_time_features, wind_direction_one_hot  # noqa: E402
from api.schemas import AnomalyResponse, ForecastResponse, Reading  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_anomaly_bundle = None
_forecast_bundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models once when the service starts."""
    global _anomaly_bundle, _forecast_bundle
    anomaly_path = MODELS_DIR / "isolation_forest.joblib"
    forecast_path = MODELS_DIR / "xgboost_forecast.joblib"

    if not anomaly_path.exists() or not forecast_path.exists():
        raise RuntimeError(
            "Model files not found. Run src/train_anomaly_model.py and "
            "src/train_forecast_models.py before starting the API."
        )

    _anomaly_bundle = joblib.load(anomaly_path)
    _forecast_bundle = joblib.load(forecast_path)
    yield


app = FastAPI(
    title="Air Quality Anomaly Detection & Forecasting API",
    description="Serves an Isolation Forest anomaly detector and an XGBoost "
                 "24-hour-ahead PM2.5 forecaster, trained on the Beijing "
                 "PM2.5 dataset (DSA8401 capstone project).",
    version="1.1.0",
    lifespan=lifespan,
)


def _build_feature_row(reading: Reading, feature_columns: list[str]) -> pd.DataFrame:
    base = reading.model_dump()
    base.update(cyclical_time_features(reading.hour, reading.month))
    base.update(wind_direction_one_hot(reading.wind_dir))

    row = {col: base.get(col) for col in feature_columns}
    missing = [c for c in feature_columns if row[c] is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Reading is missing required fields for this model: {missing}",
        )
    return pd.DataFrame([row])


@app.get("/")
def root():
    return {
        "service": "Air Quality Anomaly Detection & Forecasting API",
        "endpoints": ["/anomaly-check", "/forecast", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": _anomaly_bundle is not None and _forecast_bundle is not None}


@app.post("/anomaly-check", response_model=AnomalyResponse)
def anomaly_check(reading: Reading):
    feature_columns = _anomaly_bundle["feature_columns"]
    model = _anomaly_bundle["model"]

    # The Isolation Forest was fitted on a plain array, so pass values in the
    # same column order it was trained on.
    X = _build_feature_row(reading, feature_columns).values
    raw_pred = model.predict(X)[0]
    score = float(model.decision_function(X)[0])
    is_anomalous = raw_pred == -1

    interpretation = (
        "This reading is unusual relative to the training data. In evaluation "
        "the detector mainly flagged sustained very high PM2.5 and rarely "
        "flagged sudden one-hour jumps."
        if is_anomalous
        else "This reading is consistent with the model's learned normal range."
    )

    return AnomalyResponse(
        is_anomalous=is_anomalous,
        anomaly_score=round(score, 4),
        interpretation=interpretation,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(reading: Reading):
    feature_columns = _forecast_bundle["feature_columns"]
    model = _forecast_bundle["model"]

    X = _build_feature_row(reading, feature_columns)
    predicted = float(model.predict(X)[0])

    return ForecastResponse(
        forecast_horizon_hours=24,
        predicted_pm25=round(predicted, 1),
        origin_timestamp=reading.timestamp,
    )
