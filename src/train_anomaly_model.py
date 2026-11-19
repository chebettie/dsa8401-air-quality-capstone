"""
Trains an Isolation Forest to flag anomalous PM2.5 readings.

Isolation Forest is used because:
  - It is unsupervised: we have no pre-labelled "this was an anomaly" column,
    only raw sensor readings, so a supervised classifier isn't an option.
  - It works natively on continuous, high-dimensional tabular data without
    requiring a predefined distribution assumption (unlike a simple z-score
    or control-chart approach, which assumes near-normal data).
  - It scales linearly and trains fast, which matters given the pipeline
    will eventually need to retrain periodically as new data arrives.

Run from the project root:
    python src/train_anomaly_model.py
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLUMNS = [
    "dew_point", "temp", "pressure", "wind_speed", "snow_hours", "rain_hours",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "wind_dir_NE", "wind_dir_NW", "wind_dir_SE", "wind_dir_cv",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h",
    "pm25_roll_mean_6h", "pm25_roll_std_6h",
    "pm25_roll_mean_24h", "pm25_roll_std_24h",
    "pm25",
]


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # Drop rows with no PM2.5 reading at all (can't judge "anomalous" with
    # no target value) and rows where lag/rolling features are still NaN
    # (the first 24 hours of the series, before enough history exists).
    df = df.dropna(subset=["pm25"] + FEATURE_COLUMNS).reset_index(drop=True)
    return df


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.02) -> IsolationForest:
    X = df[FEATURE_COLUMNS].values
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/pm25_features.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/isolation_forest.joblib"))
    parser.add_argument("--flags-out", type=Path, default=Path("data/processed/anomaly_flags.csv"))
    parser.add_argument("--contamination", type=float, default=0.02,
                         help="Expected proportion of anomalous readings (prior assumption)")
    args = parser.parse_args()

    df = load_features(args.input)
    model = train_isolation_forest(df, contamination=args.contamination)

    # -1 = anomaly, 1 = normal (sklearn convention) -> convert to 1/0 flag
    raw_pred = model.predict(df[FEATURE_COLUMNS].values)
    df["anomaly_flag"] = (raw_pred == -1).astype(int)
    df["anomaly_score"] = model.decision_function(df[FEATURE_COLUMNS].values)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, args.model_out)

    args.flags_out.parent.mkdir(parents=True, exist_ok=True)
    df[["timestamp", "pm25", "anomaly_flag", "anomaly_score"]].to_csv(args.flags_out, index=False)

    n_anomalies = df["anomaly_flag"].sum()
    print(f"Rows used for training: {len(df):,}")
    print(f"Flagged anomalies: {n_anomalies:,} ({n_anomalies / len(df) * 100:.2f}%)")
    print(f"Model saved to: {args.model_out}")
    print(f"Flags saved to: {args.flags_out}")

    # Sanity check: what do the top 5 most anomalous readings look like?
    top5 = df.nsmallest(5, "anomaly_score")[["timestamp", "pm25", "anomaly_score"]]
    print("\nTop 5 most anomalous readings:")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
