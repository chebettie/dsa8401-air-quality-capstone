"""
Longer backtest of the 24h-ahead XGBoost forecast against simple baselines.

The SARIMA-vs-XGBoost comparison in train_forecast_models.py uses only 15
forecast origins, which is too few to support a firm claim. This script
trains XGBoost on everything before a one-year holdout and scores it on
every hourly origin in that year, alongside two naive baselines:
  - persistence: "PM2.5 in 24h will equal PM2.5 now"
  - training mean: "PM2.5 in 24h will equal the long-run average"

It also scores a decision-style question: does the forecast correctly
anticipate hours above an illustrative advisory threshold, and how do
precision, recall and alert burden change with the alert cut-off?

Leakage control: the last 24 hours before the holdout are purged from
training, because their 24h-ahead targets fall inside the holdout.

Results are written to data/processed/backtest_results.json so the report
can quote them without retyping.

Run from the project root:
    python src/backtest_forecast.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from train_forecast_models import FORECAST_FEATURE_COLUMNS, HORIZON_HOURS, load_data

THRESHOLD = 150.0  # illustrative advisory threshold, micrograms/m3 (an assumption)
CUTOFFS = (150, 120, 100, 80)
HOLDOUT_DAYS = 365
RESULTS_PATH = Path("data/processed/backtest_results.json")


def main():
    df = load_data("data/processed/pm25_features.csv")
    df["target"] = df["pm25"].shift(-HORIZON_HOURS)

    holdout_start = df.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    needed = ["target", "pm25"] + FORECAST_FEATURE_COLUMNS
    train_cutoff = holdout_start - pd.Timedelta(hours=HORIZON_HOURS)
    train = df[df.index < train_cutoff].dropna(subset=needed)
    test = df[df.index >= holdout_start].dropna(subset=needed)

    model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(train[FORECAST_FEATURE_COLUMNS], train["target"])

    preds = {
        "XGBoost": model.predict(test[FORECAST_FEATURE_COLUMNS]),
        "Persistence (PM2.5 now)": test["pm25"].values,
        "Training mean": np.full(len(test), train["target"].mean()),
    }
    y = test["target"].values
    actual_high = y > THRESHOLD

    out = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "holdout_start": str(test.index.min()),
        "holdout_end": str(test.index.max()),
        "test_mean_pm25": float(y.mean()),
        "test_median_pm25": float(np.median(y)),
        "threshold": THRESHOLD,
        "hours_above_threshold": int(actual_high.sum()),
        "share_hours_above_threshold": float(actual_high.mean()),
        "models": {},
        "xgboost_cutoffs": [],
    }

    print(f"Train rows: {len(train):,} | Test rows (hourly origins): {len(test):,}")
    print(f"Holdout: {test.index.min()} to {test.index.max()}")
    print(f"Mean PM2.5 in holdout targets: {y.mean():.1f} | median: {np.median(y):.1f}\n")

    print(f"{'Model':28s} {'MAE':>7s} {'RMSE':>7s}  {'Precision':>9s} {'Recall':>7s}")
    for name, p in preds.items():
        mae = mean_absolute_error(y, p)
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        pred_high = p > THRESHOLD
        tp = int((pred_high & actual_high).sum())
        precision = tp / pred_high.sum() if pred_high.sum() else float("nan")
        recall = tp / actual_high.sum() if actual_high.sum() else float("nan")
        print(f"{name:28s} {mae:7.1f} {rmse:7.1f}  {precision:9.2f} {recall:7.2f}")
        out["models"][name] = {"mae": float(mae), "rmse": rmse,
                               "precision": None if np.isnan(precision) else float(precision),
                               "recall": float(recall)}

    print(f"\nHours actually above {THRESHOLD:.0f}: {actual_high.sum():,} "
          f"({actual_high.mean() * 100:.1f}% of test hours)")

    print("\nXGBoost alert cut-offs (alert when forecast exceeds cut-off):")
    print(f"{'Cut-off':>8s} {'Precision':>10s} {'Recall':>7s} {'FalseShare':>11s} {'HoursUnderAlert':>16s}")
    p = preds["XGBoost"]
    for c in CUTOFFS:
        alert = p > c
        tp = int((alert & actual_high).sum())
        precision = tp / alert.sum()
        recall = tp / actual_high.sum()
        row = {"cutoff": c, "precision": float(precision), "recall": float(recall),
               "false_share": float(1 - precision), "alert_hours": int(alert.sum()),
               "share_hours_under_alert": float(alert.mean())}
        out["xgboost_cutoffs"].append(row)
        print(f"{c:>8d} {precision:10.2f} {recall:7.2f} {1 - precision:11.2f} {alert.mean() * 100:15.1f}%")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
