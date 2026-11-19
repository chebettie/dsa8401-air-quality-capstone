"""
Streamlit interface for the air quality anomaly detection and forecasting API.

The app sends a reading to the FastAPI service and shows the 24-hour-ahead
forecast, the anomaly check, and (for sample data) what was actually observed
24 hours later.

Run locally (with the API already running on port 8000):
    streamlit run app/streamlit_app.py
"""

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_readings.csv"

PAYLOAD_FIELDS = [
    "hour", "month", "pm25", "dew_point", "temp", "pressure", "wind_speed",
    "snow_hours", "rain_hours", "wind_dir", "pm25_lag_1h", "pm25_lag_3h",
    "pm25_lag_24h", "pm25_roll_mean_6h", "pm25_roll_std_6h",
    "pm25_roll_mean_24h", "pm25_roll_std_24h",
]

st.set_page_config(page_title="Air quality forecast and anomaly check", layout="wide")
st.title("Air quality: 24-hour forecast and anomaly check")
st.caption(
    "DSA 8401 capstone demonstration. The models were trained on hourly PM2.5 data from the "
    "US Embassy monitor in Beijing, 2010 to 2014, and are not calibrated for any other location."
)


@st.cache_data
def load_sample() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


if not SAMPLE_PATH.exists():
    st.error(f"Sample data not found at {SAMPLE_PATH}. Run `python src/make_sample_data.py`.")
    st.stop()

sample = load_sample()

with st.sidebar:
    st.header("Reading")
    st.write(
        "Pick an hour from the final 14 days of 2014. Those days were held out when the "
        "forecast model was trained. The anomaly model is unsupervised and saw all of the data."
    )
    chosen = st.selectbox(
        "Forecast origin",
        options=sample["timestamp"],
        index=len(sample) // 2,
        format_func=lambda t: t.strftime("%d %b %Y, %H:%M"),
    )
    row = sample[sample["timestamp"] == chosen].iloc[0]
    edit = st.checkbox("Edit the inputs before sending", value=False)

    values = {}
    if edit:
        values["pm25"] = st.number_input("PM2.5 now (µg/m³)", 0.0, 1000.0, float(row["pm25"]))
        values["pm25_lag_1h"] = st.number_input("PM2.5 1 hour ago", 0.0, 1000.0, float(row["pm25_lag_1h"]))
        values["pm25_lag_3h"] = st.number_input("PM2.5 3 hours ago", 0.0, 1000.0, float(row["pm25_lag_3h"]))
        values["pm25_lag_24h"] = st.number_input("PM2.5 24 hours ago", 0.0, 1000.0, float(row["pm25_lag_24h"]))
        values["temp"] = st.number_input("Temperature (°C)", -40.0, 50.0, float(row["temp"]))
        values["wind_speed"] = st.number_input("Cumulated wind speed (m/s)", 0.0, 600.0, float(row["wind_speed"]))
        values["wind_dir"] = st.selectbox(
            "Wind direction", ["NE", "NW", "SE", "cv"], index=["NE", "NW", "SE", "cv"].index(row["wind_dir"])
        )

payload = {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in PAYLOAD_FIELDS}
payload.update(values)
payload["timestamp"] = chosen.isoformat()

if st.button("Run forecast and anomaly check", type="primary"):
    try:
        forecast = requests.post(f"{API_URL}/forecast", json=payload, timeout=15)
        anomaly = requests.post(f"{API_URL}/anomaly-check", json=payload, timeout=15)
        forecast.raise_for_status()
        anomaly.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}. Start it first. Details: {exc}")
        st.stop()

    f, a = forecast.json(), anomaly.json()
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast PM2.5 in 24 hours (µg/m³)", f"{f['predicted_pm25']:.0f}")
    c2.metric("Observed 24 hours later (µg/m³)", f"{row['pm25_24h_later']:.0f}" if not edit else "n/a (inputs edited)")
    c3.metric("Anomaly check", "Anomalous" if a["is_anomalous"] else "Within normal range")

    st.write(a["interpretation"])
    st.caption(f"Anomaly score {a['anomaly_score']} (lower scores are more anomalous).")

    history = sample[(sample["timestamp"] <= chosen) & (sample["timestamp"] > chosen - pd.Timedelta(hours=72))]
    chart = history.set_index("timestamp")[["pm25"]].rename(columns={"pm25": "Observed PM2.5"})
    target_time = chosen + pd.Timedelta(hours=24)
    chart.loc[target_time, "Forecast for +24 h"] = f["predicted_pm25"]
    if not edit:
        chart.loc[target_time, "Observed at +24 h"] = row["pm25_24h_later"]
    st.line_chart(chart.sort_index())

    st.info(
        "On the 2014 test year the forecast was only modestly more accurate than assuming PM2.5 "
        "stays at its current level, and alerting at 150 µg/m³ caught about three in ten of the hours above "
        "that level. Treat a single forecast as a rough guide."
    )
