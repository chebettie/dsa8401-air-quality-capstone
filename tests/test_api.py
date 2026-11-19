"""
Smoke tests for the API. They need the trained model files in models/.

Run from the project root:
    pytest -q
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

VALID = {
    "timestamp": "2014-12-30T12:00:00", "hour": 12, "month": 12,
    "pm25": 180.0, "dew_point": -5.0, "temp": 2.0, "pressure": 1020.0,
    "wind_speed": 3.5, "snow_hours": 0, "rain_hours": 0, "wind_dir": "NW",
    "pm25_lag_1h": 175.0, "pm25_lag_3h": 160.0, "pm25_lag_24h": 140.0,
    "pm25_roll_mean_6h": 165.0, "pm25_roll_std_6h": 12.0,
    "pm25_roll_mean_24h": 150.0, "pm25_roll_std_24h": 20.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] is True


def test_forecast_returns_a_number(client):
    r = client.post("/forecast", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["forecast_horizon_hours"] == 24
    assert body["predicted_pm25"] >= 0


def test_anomaly_check_returns_flag_and_score(client):
    r = client.post("/anomaly-check", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["is_anomalous"], bool)
    assert isinstance(body["anomaly_score"], float)


@pytest.mark.parametrize("change", [
    {"pm25": -5},                 # negative concentration
    {"wind_dir": "XYZ"},          # unknown wind direction
    {"hour": 3},                  # hour disagrees with timestamp
    {"timestamp": "not-a-date"},
])
def test_invalid_readings_are_rejected(client, change):
    r = client.post("/forecast", json={**VALID, **change})
    assert r.status_code == 422


def test_missing_field_is_rejected(client):
    body = {k: v for k, v in VALID.items() if k != "pm25_lag_1h"}
    assert client.post("/anomaly-check", json=body).status_code == 422
