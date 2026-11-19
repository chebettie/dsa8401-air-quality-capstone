"""
Writes a small sample of engineered readings for the Streamlit demo.

Takes the last 14 days of the processed feature table (the window held out from
the deployed forecast model's training), keeps only complete
rows, and keeps only rows whose reading 24 hours later also exists, so the
interface can show the forecast next to what was actually observed.

Run from the project root:
    python src/make_sample_data.py
"""

from pathlib import Path

import pandas as pd

REQUIRED = [
    "pm25", "dew_point", "temp", "pressure", "wind_speed", "snow_hours", "rain_hours",
    "wind_dir", "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h",
    "pm25_roll_mean_6h", "pm25_roll_std_6h", "pm25_roll_mean_24h", "pm25_roll_std_24h",
]


def main(days: int = 14):
    df = pd.read_csv("data/processed/pm25_features.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df["pm25_24h_later"] = df["pm25"].shift(-24)

    recent = df[df.index >= df.index.max() - pd.Timedelta(days=days)]
    recent = recent.dropna(subset=REQUIRED + ["pm25_24h_later"])

    # Keep a short history column set for the chart as well.
    cols = ["hour", "month"] + REQUIRED + ["pm25_24h_later"]
    out = recent[cols].reset_index()
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/sample/sample_readings.csv", index=False)
    print(f"Wrote {len(out)} rows to data/sample/sample_readings.csv "
          f"({out['timestamp'].min()} to {out['timestamp'].max()})")


if __name__ == "__main__":
    main()
