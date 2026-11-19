"""
Tests whether an LSTM improves on XGBoost for the 24-hour-ahead forecast.

The proposal stated that an LSTM was not expected to be necessary and that
this "will be tested rather than assumed". This script is that test. The
LSTM is an evaluation result only: it is not deployed.

Design:
  - Input: the 48 hourly steps up to and including the forecast origin.
    Each step holds PM2.5, a flag marking whether that PM2.5 value was
    missing, the weather variables, wind direction and the cyclical time
    encodings. Nothing after the origin is used.
  - Missing PM2.5 inside an input window is linearly interpolated so the
    network receives a complete sequence, and the missing flag tells the
    network which values were filled. Targets are never interpolated: only
    observed readings are used for training and scoring.
  - Target: the observed PM2.5 reading 24 hours after the origin.
  - Split: the same one-year holdout as backtest_forecast.py, with the same
    24-hour purge. The last 10% of the training period (by time) is used as
    a validation set for early stopping, with a further 24-hour purge.
  - Scaling: inputs and target are standardised with training-set statistics
    only.
  - Comparison: the LSTM, XGBoost and persistence are scored on the identical
    set of holdout origins.
  - The LSTM is trained with three random seeds and the spread is reported,
    because a single neural-network run can be lucky or unlucky.
  - The gap between XGBoost and the LSTM is given with a 95% interval from a
    weekly block bootstrap (see eval_utils.py), because hourly errors are
    autocorrelated.

Neither model is tuned beyond sensible defaults, so this is a like-for-like
check under a small budget, not a claim about the best achievable LSTM.

Results are written to data/processed/lstm_results.json.

Run from the project root:
    python src/backtest_lstm.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from xgboost import XGBRegressor

from eval_utils import block_bootstrap_mae_diff
from train_forecast_models import FORECAST_FEATURE_COLUMNS, HORIZON_HOURS, load_data

HOLDOUT_DAYS = 365
WINDOW = 48          # hours of history fed to the LSTM
HIDDEN = 64
MAX_EPOCHS = 30
PATIENCE = 4
BATCH = 256
SEEDS = (0, 1, 2)
RESULTS_PATH = Path("data/processed/lstm_results.json")

SEQ_COLUMNS = [
    "pm25_filled", "pm25_missing",
    "dew_point", "temp", "pressure", "wind_speed", "snow_hours", "rain_hours",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "wind_dir_NE", "wind_dir_NW", "wind_dir_SE", "wind_dir_cv",
]
SCALED = ["pm25_filled", "dew_point", "temp", "pressure", "wind_speed", "snow_hours", "rain_hours"]


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def windows(values: np.ndarray, positions: np.ndarray) -> torch.Tensor:
    """Stack the WINDOW rows ending at each origin position."""
    idx = positions[:, None] - np.arange(WINDOW - 1, -1, -1)[None, :]
    return torch.tensor(values[idx], dtype=torch.float32)


def train_one(seed, X_tr, y_tr, X_va, y_va):
    torch.manual_seed(seed)
    model = LSTMForecaster(X_tr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(X_tr), generator=gen)
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(X_tr[b]), y_tr[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_va), y_va).item()
        if val < best - 1e-4:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, epoch + 1


def score(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def main():
    torch.set_num_threads(2)
    df = load_data("data/processed/pm25_features.csv")
    df["target"] = df["pm25"].shift(-HORIZON_HOURS)
    df["pm25_filled"] = df["pm25"].interpolate(method="linear", limit_direction="both")
    df["pos"] = np.arange(len(df))

    holdout_start = df.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train_cutoff = holdout_start - pd.Timedelta(hours=HORIZON_HOURS)
    needed = ["target", "pm25"] + FORECAST_FEATURE_COLUMNS

    # Same origin sets as backtest_forecast.py, minus the first WINDOW rows.
    eligible = df[df["pos"] >= WINDOW - 1].dropna(subset=needed)
    train_all = eligible[eligible.index < train_cutoff]
    test = eligible[eligible.index >= holdout_start]

    # Chronological validation split inside the training period, purged.
    val_start = train_all.index[int(len(train_all) * 0.9)]
    fit = train_all[train_all.index < val_start - pd.Timedelta(hours=HORIZON_HOURS)]
    val = train_all[train_all.index >= val_start]

    # Standardise with statistics from the fitting period only.
    fit_rows = df[df.index < fit.index.max()]
    mu, sd = fit_rows[SCALED].mean(), fit_rows[SCALED].std()
    seq = df[SEQ_COLUMNS].copy().astype(float)
    seq[SCALED] = (seq[SCALED] - mu) / sd
    values = seq.values
    y_mu, y_sd = fit["target"].mean(), fit["target"].std()

    def tensors(frame):
        return (windows(values, frame["pos"].values),
                torch.tensor(((frame["target"] - y_mu) / y_sd).values, dtype=torch.float32))

    X_fit, y_fit = tensors(fit)
    X_val, y_val = tensors(val)
    X_test, _ = tensors(test)
    y_test = test["target"].values

    print(f"Fit rows: {len(fit):,} | validation rows: {len(val):,} | test origins: {len(test):,}")
    print(f"Holdout: {test.index.min()} to {test.index.max()}\n")

    runs, lstm_abs_err = [], []
    for seed in SEEDS:
        t0 = time.time()
        model, epochs = train_one(seed, X_fit, y_fit, X_val, y_val)
        with torch.no_grad():
            pred = model(X_test).numpy() * y_sd + y_mu
        s = score(y_test, pred)
        lstm_abs_err.append(np.abs(y_test - pred))
        s.update({"seed": seed, "epochs": epochs})
        runs.append(s)
        print(f"LSTM seed {seed}: MAE {s['mae']:.1f}  RMSE {s['rmse']:.1f}  "
              f"({epochs} epochs, {time.time() - t0:.0f}s)")

    # XGBoost trained on the full training period, as in backtest_forecast.py.
    xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(train_all[FORECAST_FEATURE_COLUMNS], train_all["target"])
    xgb_pred = xgb.predict(test[FORECAST_FEATURE_COLUMNS])
    xgb_score = score(y_test, xgb_pred)

    # Is the LSTM-XGBoost gap larger than the noise? Positive = LSTM better.
    # The LSTM error at each origin is averaged over the three seeds.
    gap = block_bootstrap_mae_diff(test.index, np.abs(y_test - xgb_pred), np.mean(lstm_abs_err, axis=0))
    pers_score = score(y_test, test["pm25"].values)

    maes, rmses = [r["mae"] for r in runs], [r["rmse"] for r in runs]
    out = {
        "holdout_start": str(test.index.min()), "holdout_end": str(test.index.max()),
        "fit_rows": int(len(fit)), "validation_rows": int(len(val)), "test_rows": int(len(test)),
        "window_hours": WINDOW, "hidden_units": HIDDEN,
        "lstm_runs": runs,
        "lstm_mean": {"mae": float(np.mean(maes)), "rmse": float(np.mean(rmses))},
        "lstm_range": {"mae": [float(min(maes)), float(max(maes))],
                       "rmse": [float(min(rmses)), float(max(rmses))]},
        "xgboost": xgb_score,
        "persistence": pers_score,
        "xgboost_minus_lstm_mae": gap,
    }
    print(f"\nLSTM mean over {len(SEEDS)} seeds: MAE {out['lstm_mean']['mae']:.1f}  RMSE {out['lstm_mean']['rmse']:.1f}")
    print(f"XGBoost:                 MAE {xgb_score['mae']:.1f}  RMSE {xgb_score['rmse']:.1f}")
    print(f"Persistence:             MAE {pers_score['mae']:.1f}  RMSE {pers_score['rmse']:.1f}")

    print(f"XGBoost MAE minus LSTM MAE: {gap['mae_difference']:.2f} "
          f"(95% interval {gap['ci95'][0]:.2f} to {gap['ci95'][1]:.2f}, weekly block bootstrap)")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
