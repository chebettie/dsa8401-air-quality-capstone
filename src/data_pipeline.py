"""
Data pipeline for the Beijing PM2.5 Air Quality capstone project.

Stages:
  1. Ingest raw hourly CSV
  2. Clean / standardise (timestamps, column names, categorical encoding)
  3. Flag (not silently impute) missing PM2.5 readings
  4. Engineer time-based features (lags, rolling stats, cyclical time encodings)
  5. Write a processed parquet/csv ready for modelling

Run from the project root:
    python src/data_pipeline.py --input data/raw/PRSA_data_2010_1_1-2014_12_31.csv \
                                 --output data/processed/pm25_features.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_COLUMNS = {
    "No": "row_id",
    "year": "year",
    "month": "month",
    "day": "day",
    "hour": "hour",
    "pm2.5": "pm25",
    "DEWP": "dew_point",
    "TEMP": "temp",
    "PRES": "pressure",
    "cbwd": "wind_dir",
    "Iws": "wind_speed",
    "Is": "snow_hours",
    "Ir": "rain_hours",
}


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=RAW_COLUMNS)
    return df


def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].astype(str).agg("-".join, axis=1),
        format="%Y-%m-%d-%H",
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def flag_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Flag missing PM2.5 readings rather than silently imputing them."""
    df["pm25_missing"] = df["pm25"].isna().astype(int)
    return df


def encode_wind_direction(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encode categorical wind direction (NE, NW, SE, cv = calm/variable)
    dummies = pd.get_dummies(df["wind_dir"], prefix="wind_dir", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Hour-of-day and month-of-year as sine/cosine pairs so the model sees
    # e.g. hour 23 and hour 0 as close together, not far apart.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # Lag features: PM2.5 1h, 3h, 24h ago
    for lag in (1, 3, 24):
        df[f"pm25_lag_{lag}h"] = df["pm25"].shift(lag)

    # Rolling mean / std over the past 6h and 24h (shifted so we never leak
    # the current reading into its own rolling window)
    for window in (6, 24):
        df[f"pm25_roll_mean_{window}h"] = (
            df["pm25"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"pm25_roll_std_{window}h"] = (
            df["pm25"].shift(1).rolling(window=window, min_periods=1).std()
        )
    return df


def run_pipeline(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_raw(input_path)
    df = build_timestamp(df)
    df = flag_missing(df)
    df = encode_wind_direction(df)
    df = add_cyclical_time_features(df)
    df = add_lag_and_rolling_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Run the PM2.5 data pipeline")
    parser.add_argument("--input", type=Path, default=Path("data/raw/PRSA_data_2010_1_1-2014_12_31.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/pm25_features.csv"))
    args = parser.parse_args()

    df = run_pipeline(args.input, args.output)

    print(f"Rows: {len(df):,}")
    print(f"Missing PM2.5 before feature engineering: {df['pm25_missing'].sum():,} "
          f"({df['pm25_missing'].mean() * 100:.1f}%)")
    print(f"Columns produced: {len(df.columns)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
