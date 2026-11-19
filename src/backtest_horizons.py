"""
Backtest of the XGBoost forecast at 24 and 48 hours ahead.

The proposal committed to short-term forecasts of 24 to 48 hours. The other
scripts evaluate 24 hours only. This script repeats the one-year holdout
backtest of backtest_forecast.py for both horizons, so the report can state
how much accuracy is lost when the horizon doubles.

Protocol, identical for both horizons:
  - Direct forecasting: a separate model is trained for each horizon, with
    the reading h hours after the origin as the target. No recursive
    forecasts are fed back in as inputs.
  - Features are the same as the deployed model: everything observed at or
    before the origin. No weather from the forecast period is used, because
    it would not be known at prediction time.
  - Holdout: the final 365 days. The last h hours before the holdout are
    purged from training, because their targets fall inside the holdout.
  - Baselines: persistence (the reading at the origin) and the training mean.
  - Both horizons are scored on the same set of origins (those where every
    feature and both targets are observed), so the two rows are comparable.

Only the 24-hour model is deployed in the API. The 48-hour model is an
evaluation result, not a served artifact.

Results are written to data/processed/horizon_results.json.

Run from the project root:
    python src/backtest_horizons.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from eval_utils import block_bootstrap_mae_diff
from train_forecast_models import FORECAST_FEATURE_COLUMNS, load_data

HORIZONS = (24, 48)
HOLDOUT_DAYS = 365
THRESHOLD = 150.0  # same illustrative advisory threshold as backtest_forecast.py
RESULTS_PATH = Path("data/processed/horizon_results.json")


def metrics(y, p):
    actual_high, pred_high = y > THRESHOLD, p > THRESHOLD
    tp = int((actual_high & pred_high).sum())
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "precision": float(tp / pred_high.sum()) if pred_high.sum() else None,
        "recall": float(tp / actual_high.sum()) if actual_high.sum() else None,
    }


def main():
    df = load_data("data/processed/pm25_features.csv")
    for h in HORIZONS:
        df[f"target_{h}h"] = df["pm25"].shift(-h)

    holdout_start = df.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    targets = [f"target_{h}h" for h in HORIZONS]
    # Same test origins for every horizon.
    test = df[df.index >= holdout_start].dropna(subset=["pm25"] + targets + FORECAST_FEATURE_COLUMNS)

    out = {"holdout_start": str(test.index.min()), "holdout_end": str(test.index.max()),
           "test_rows": int(len(test)), "threshold": THRESHOLD, "horizons": {}}

    print(f"Holdout origins: {len(test):,} ({test.index.min()} to {test.index.max()})\n")
    print(f"{'Horizon':>8s} {'Model':14s} {'MAE':>7s} {'RMSE':>7s} {'Precision':>10s} {'Recall':>7s}")

    for h in HORIZONS:
        target = f"target_{h}h"
        train_cutoff = holdout_start - pd.Timedelta(hours=h)
        train = df[df.index < train_cutoff].dropna(subset=["pm25", target] + FORECAST_FEATURE_COLUMNS)

        model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(train[FORECAST_FEATURE_COLUMNS], train[target])

        y = test[target].values
        preds = {
            "XGBoost": model.predict(test[FORECAST_FEATURE_COLUMNS]),
            "Persistence": test["pm25"].values,
            "Training mean": np.full(len(test), train[target].mean()),
        }
        res = {"train_rows": int(len(train)), "models": {}}
        for name, p in preds.items():
            m = metrics(y, p)
            res["models"][name] = m
            prec = f"{m['precision']:.2f}" if m["precision"] is not None else "n/a"
            print(f"{h:>7d}h {name:14s} {m['mae']:7.1f} {m['rmse']:7.1f} {prec:>10s} {m['recall']:7.2f}")
        xgb, per = res["models"]["XGBoost"], res["models"]["Persistence"]
        res["xgboost_mae_improvement_over_persistence"] = float(1 - xgb["mae"] / per["mae"])
        # Uncertainty on the gaps (positive = XGBoost better), weekly block bootstrap.
        e = {k: np.abs(y - p) for k, p in preds.items()}
        res["persistence_minus_xgboost_mae"] = block_bootstrap_mae_diff(test.index, e["Persistence"], e["XGBoost"])
        res["training_mean_minus_xgboost_mae"] = block_bootstrap_mae_diff(test.index, e["Training mean"], e["XGBoost"])
        for label in ("persistence_minus_xgboost_mae", "training_mean_minus_xgboost_mae"):
            g = res[label]
            print(f"         {label}: {g['mae_difference']:.1f} (95% interval {g['ci95'][0]:.1f} to {g['ci95'][1]:.1f})")
        out["horizons"][f"{h}h"] = res
        print()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
