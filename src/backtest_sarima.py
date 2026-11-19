"""
Extended SARIMA vs XGBoost vs persistence comparison over many forecast origins.

train_forecast_models.py compares SARIMA and XGBoost on only 15 forecast
origins. This script repeats the comparison with one forecast origin per day
over a longer window (default 90 days), on the same origins for all three
methods, and scores every method against observed readings only.

SARIMA is fit once on a rolling 120-day window and its state is rolled
forward with apply() rather than refitted at each origin (refitting a
seasonal model daily is impractical in this scope), so its parameters are
fixed from the first origin. This favours XGBoost slightly less than a
fully refitted SARIMA would, and is stated as a limitation in the report.

Run from the project root:
    python src/backtest_sarima.py --days 90
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from train_forecast_models import (
    HORIZON_HOURS, build_continuous_series, evaluate_sarima,
    evaluate_xgboost, load_data, prepare_xgb_training_table,
)


def score(df):
    return {"MAE": float(mean_absolute_error(df["actual"], df["predicted"])),
            "RMSE": float(np.sqrt(mean_squared_error(df["actual"], df["predicted"]))),
            "n": int(len(df))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--input", type=Path, default=Path("data/processed/pm25_features.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/sarima_extended_results.json"))
    args = parser.parse_args()

    df = load_data(args.input)
    series = build_continuous_series(df)

    end = df.index.max() - pd.Timedelta(hours=HORIZON_HOURS)
    start = end - pd.Timedelta(days=args.days)
    origins = pd.date_range(start, end, freq="24h")

    t0 = time.time()
    sarima = evaluate_sarima(series, origins, df["pm25"])
    print(f"SARIMA done in {time.time() - t0:.0f}s on {len(sarima)} origins")

    table = prepare_xgb_training_table(df[df.index < start - pd.Timedelta(hours=HORIZON_HOURS)])
    _, xgb = evaluate_xgboost(table, origins, df)

    persistence = []
    for o in origins:
        target = o + pd.Timedelta(hours=HORIZON_HOURS)
        now, actual = df["pm25"].get(o, np.nan), df["pm25"].reindex([target]).iloc[0]
        if pd.notna(now) and pd.notna(actual):
            persistence.append({"origin": o, "predicted": now, "actual": actual})
    persistence = pd.DataFrame(persistence)

    # Score all three on the identical set of origins.
    common = set(sarima["origin"]) & set(xgb["origin"]) & set(persistence["origin"])
    res = {}
    for name, d in (("SARIMA", sarima), ("XGBoost", xgb), ("Persistence", persistence)):
        res[name] = score(d[d["origin"].isin(common)])
        print(f"{name:12s} MAE {res[name]['MAE']:6.1f}  RMSE {res[name]['RMSE']:6.1f}  n={res[name]['n']}")

    args.out.write_text(json.dumps({"window_start": str(start), "window_end": str(end),
                                    "origins": len(common), "results": res}, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
