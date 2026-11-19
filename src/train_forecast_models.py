"""
Compares a classical statistical baseline (SARIMA) against a gradient-
boosted model (XGBoost) for 24-hour-ahead PM2.5 forecasting.

Evaluation protocol: "rolling origin, 24h horizon".
  For a series of forecast origins across the held-out test period, each
  model predicts the PM2.5 value 24 hours after that origin, using only
  information available AT the origin (no future leakage). Both models are
  scored on the same origins so the comparison is apples-to-apples.

Why SARIMA is restricted to a recent window:
  Fitting a seasonal ARIMA with a 24-hour seasonal period on ~40,000 hourly
  points is computationally impractical for this scope (state-space
  estimation cost grows quickly with series length and seasonal order).
  SARIMA is therefore fit on the most recent ~120 days of history. This is
  a compute-driven choice and is listed as a limitation in the report: a
  longer window or a tuned order might do better. XGBoost, by contrast, is trained on the full available history
  since tree-based models scale comfortably to this size.

Run from the project root:
    python src/train_forecast_models.py
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

HORIZON_HOURS = 24
SARIMA_WINDOW_DAYS = 120
TEST_DAYS = 14
ROLL_STEP_HOURS = 24  # one forecast origin per day, to keep SARIMA tractable

FORECAST_FEATURE_COLUMNS = [
    "dew_point", "temp", "pressure", "wind_speed", "snow_hours", "rain_hours",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "wind_dir_NE", "wind_dir_NW", "wind_dir_SE", "wind_dir_cv",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h",
    "pm25_roll_mean_6h", "pm25_roll_std_6h",
    "pm25_roll_mean_24h", "pm25_roll_std_24h",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


def build_continuous_series(df: pd.DataFrame) -> pd.Series:
    """Reindex to a complete hourly index and linearly interpolate gaps.

    SARIMA requires a complete, evenly spaced series. Interpolation here is
    used only to give the statistical model a continuous history to fit on
    -- it is a modelling necessity, not a claim that the interpolated values
    are real readings. Evaluation is always against genuine observed values.
    """
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    series = df["pm25"].reindex(full_index)
    series = series.interpolate(method="linear", limit_direction="both")
    return series


def prepare_xgb_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a direct 24h-ahead supervised learning table.

    Target at row t is the PM2.5 reading 24 hours after t. Features are
    everything observed AT or BEFORE t (lags/rolling stats are already
    shifted in the pipeline), so no future information leaks into training.
    """
    table = df.copy()
    table["target_pm25_24h"] = table["pm25"].shift(-HORIZON_HOURS)
    table = table.dropna(subset=["target_pm25_24h", "pm25"] + FORECAST_FEATURE_COLUMNS)
    return table


def evaluate_sarima(series: pd.Series, test_origins: pd.DatetimeIndex,
                    observed: pd.Series) -> pd.DataFrame:
    results = []
    window = pd.Timedelta(days=SARIMA_WINDOW_DAYS)

    model_fit = None
    for origin in test_origins:
        train_window = series.loc[origin - window: origin]
        if model_fit is None:
            model = SARIMAX(
                train_window,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 24),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = model.fit(disp=False)
        else:
            # Cheaper than refitting: carry forward fitted parameters and
            # just update the state with the newest window of observations.
            model_fit = model_fit.apply(train_window, refit=False)

        forecast = model_fit.get_forecast(steps=HORIZON_HOURS)
        predicted = forecast.predicted_mean.iloc[-1]
        # Score against the genuine observed reading, never an interpolated one.
        actual = observed.reindex([origin + pd.Timedelta(hours=HORIZON_HOURS)]).iloc[0]
        if pd.isna(actual):
            continue
        results.append({"origin": origin, "predicted": predicted, "actual": actual})

    return pd.DataFrame(results)


def evaluate_xgboost(train_table: pd.DataFrame, test_origins: pd.DatetimeIndex, df: pd.DataFrame) -> pd.DataFrame:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    X_train = train_table[FORECAST_FEATURE_COLUMNS]
    y_train = train_table["target_pm25_24h"]
    model.fit(X_train, y_train)

    results = []
    for origin in test_origins:
        if origin not in df.index:
            continue
        row = df.loc[[origin], FORECAST_FEATURE_COLUMNS]
        if row.isna().any(axis=None):
            continue
        predicted = model.predict(row)[0]
        target_time = origin + pd.Timedelta(hours=HORIZON_HOURS)
        actual = df["pm25"].reindex([target_time]).iloc[0]
        if pd.isna(actual):
            continue
        results.append({"origin": origin, "predicted": predicted, "actual": actual})

    return model, pd.DataFrame(results)


def score(results: pd.DataFrame) -> dict:
    mae = mean_absolute_error(results["actual"], results["predicted"])
    rmse = np.sqrt(mean_squared_error(results["actual"], results["predicted"]))
    return {"MAE": mae, "RMSE": rmse, "n_forecasts": len(results)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/pm25_features.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/xgboost_forecast.joblib"))
    parser.add_argument("--results-out", type=Path, default=Path("data/processed/forecast_comparison.csv"))
    args = parser.parse_args()

    df = load_data(args.input)
    series = build_continuous_series(df)

    test_end = df.index.max() - pd.Timedelta(hours=HORIZON_HOURS)
    test_start = test_end - pd.Timedelta(days=TEST_DAYS)
    test_origins = pd.date_range(test_start, test_end, freq=f"{ROLL_STEP_HOURS}h")

    print(f"Test period: {test_start} to {test_end} ({len(test_origins)} forecast origins)")

    print("\nFitting / rolling SARIMA...")
    sarima_results = evaluate_sarima(series, test_origins, df["pm25"])
    sarima_scores = score(sarima_results)
    print(f"SARIMA  -> MAE: {sarima_scores['MAE']:.2f}, RMSE: {sarima_scores['RMSE']:.2f}, "
          f"n={sarima_scores['n_forecasts']}")

    print("\nTraining XGBoost (full history, direct 24h-ahead target)...")
    xgb_train_table = prepare_xgb_training_table(
        df[df.index < test_start - pd.Timedelta(hours=HORIZON_HOURS)])
    xgb_model, xgb_results = evaluate_xgboost(xgb_train_table, test_origins, df)
    xgb_scores = score(xgb_results)
    print(f"XGBoost -> MAE: {xgb_scores['MAE']:.2f}, RMSE: {xgb_scores['RMSE']:.2f}, "
          f"n={xgb_scores['n_forecasts']}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": xgb_model, "feature_columns": FORECAST_FEATURE_COLUMNS}, args.model_out)

    sarima_results["model"] = "sarima"
    xgb_results["model"] = "xgboost"
    combined = pd.concat([sarima_results, xgb_results], ignore_index=True)
    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.results_out, index=False)

    better = "XGBoost" if xgb_scores["MAE"] < sarima_scores["MAE"] else "SARIMA"
    print(f"\nLower MAE on this test period: {better}")
    print(f"Saved XGBoost model to: {args.model_out}")
    print(f"Saved comparison results to: {args.results_out}")


if __name__ == "__main__":
    main()
