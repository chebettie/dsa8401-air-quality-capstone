from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Reading(BaseModel):
    """A single hourly reading plus enough recent history to build the
    lag/rolling features the models were trained on. In production this
    would normally be assembled automatically from a sensor stream and a
    small rolling buffer; here the caller supplies it directly so the API
    can be tested standalone.
    """

    timestamp: str = Field(..., description="ISO timestamp, e.g. 2014-12-30T12:00:00")
    hour: int = Field(..., ge=0, le=23)
    month: int = Field(..., ge=1, le=12)

    pm25: float = Field(..., ge=0, description="Current PM2.5 reading (micrograms/m3)")
    dew_point: float
    temp: float
    pressure: float
    wind_speed: float = Field(..., ge=0)
    snow_hours: float = Field(0, ge=0)
    rain_hours: float = Field(0, ge=0)
    wind_dir: Literal["NE", "NW", "SE", "cv"] = Field(..., description="One of NE, NW, SE, cv (calm/variable)")

    pm25_lag_1h: float = Field(..., ge=0)
    pm25_lag_3h: float = Field(..., ge=0)
    pm25_lag_24h: float = Field(..., ge=0)
    pm25_roll_mean_6h: float = Field(..., ge=0)
    pm25_roll_std_6h: float = Field(0, ge=0)
    pm25_roll_mean_24h: float = Field(..., ge=0)
    pm25_roll_std_24h: float = Field(0, ge=0)

    @model_validator(mode="after")
    def timestamp_matches_hour_and_month(self):
        # hour and month drive the cyclical time features, so they must agree
        # with the timestamp. Reject the request instead of guessing which is right.
        try:
            ts = datetime.fromisoformat(self.timestamp)
        except ValueError as exc:
            raise ValueError("timestamp must be ISO format, e.g. 2014-12-30T12:00:00") from exc
        if ts.hour != self.hour or ts.month != self.month:
            raise ValueError("hour and month must match the timestamp")
        return self

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "timestamp": "2014-12-30T12:00:00",
            "hour": 12,
            "month": 12,
            "pm25": 180.0,
            "dew_point": -5.0,
            "temp": 2.0,
            "pressure": 1020.0,
            "wind_speed": 3.5,
            "snow_hours": 0,
            "rain_hours": 0,
            "wind_dir": "NW",
            "pm25_lag_1h": 175.0,
            "pm25_lag_3h": 160.0,
            "pm25_lag_24h": 140.0,
            "pm25_roll_mean_6h": 165.0,
            "pm25_roll_std_6h": 12.0,
            "pm25_roll_mean_24h": 150.0,
            "pm25_roll_std_24h": 20.0,
        }
    })


class AnomalyResponse(BaseModel):
    is_anomalous: bool
    anomaly_score: float
    interpretation: str


class ForecastResponse(BaseModel):
    forecast_horizon_hours: int
    predicted_pm25: float
    origin_timestamp: str
