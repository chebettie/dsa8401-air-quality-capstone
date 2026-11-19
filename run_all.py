"""
Runs the whole reproducible pipeline from raw data to trained models,
evaluation results and figures.

    python run_all.py            # full run (a few minutes)
    python run_all.py --quick    # skips the SARIMA backtest and the LSTM test (under a minute)

Steps, in order:
  1. data_pipeline.py          raw CSV -> engineered feature table
  2. train_anomaly_model.py    Isolation Forest -> models/isolation_forest.joblib
  3. train_forecast_models.py  SARIMA vs XGBoost (15 origins) -> models/xgboost_forecast.joblib
  4. backtest_forecast.py      XGBoost vs baselines over the 2014 test year
  5. backtest_sarima.py        SARIMA vs XGBoost vs persistence, daily origins, 2014
  6. backtest_horizons.py      XGBoost vs baselines at 24 and 48 hours ahead
  7. backtest_lstm.py          LSTM vs XGBoost vs persistence (needs PyTorch)
  8. evaluate_anomaly_model.py anomaly episodes, sensitivity, spike tests
  9. make_sample_data.py       sample readings for the Streamlit demo
 10. make_plots.py             figures for the report and README
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Data pipeline", ["src/data_pipeline.py"]),
    ("Train anomaly model", ["src/train_anomaly_model.py"]),
    ("Train forecast models", ["src/train_forecast_models.py"]),
    ("Backtest XGBoost vs baselines", ["src/backtest_forecast.py"]),
    ("Backtest SARIMA vs XGBoost vs persistence", ["src/backtest_sarima.py", "--days", "365"]),
    ("Backtest 24h and 48h horizons", ["src/backtest_horizons.py"]),
    ("Backtest LSTM vs XGBoost", ["src/backtest_lstm.py"]),
    ("Evaluate anomaly model", ["src/evaluate_anomaly_model.py"]),
    ("Make sample data", ["src/make_sample_data.py"]),
    ("Make figures", ["src/make_plots.py"]),
]

SLOW_STEPS = {"src/backtest_sarima.py", "src/backtest_lstm.py"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the 365-day SARIMA backtest and the LSTM test")
    args = parser.parse_args()

    raw = ROOT / "data" / "raw" / "PRSA_data_2010_1_1-2014_12_31.csv"
    if not raw.exists():
        sys.exit(f"Raw data not found at {raw}. Download it from the UCI repository (see README).")

    for name, cmd in STEPS:
        if args.quick and cmd[0] in SLOW_STEPS:
            print(f"\n=== {name}: skipped (--quick) ===")
            continue
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"Step failed: {name}")
        print(f"--- done in {time.time() - t0:.0f}s ---")

    print("\nAll steps completed. Next: docker compose up --build")


if __name__ == "__main__":
    main()
