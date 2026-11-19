"""
Generates the figures used in the report and README from saved results.

Run from the project root, after the pipeline, training and evaluation steps:
    python src/make_plots.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("docs")
OUT.mkdir(exist_ok=True)


def anomaly_timeline():
    df = pd.read_csv("data/processed/anomaly_flags.csv", parse_dates=["timestamp"])
    flagged = df[df["anomaly_flag"] == 1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["pm25"], color="#b0b0b0", linewidth=0.4, label="Hourly PM2.5")
    ax.scatter(flagged["timestamp"], flagged["pm25"], s=5, color="#d62728", label="Flagged as anomalous")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Isolation Forest flags across 2010 to 2014 (807 flagged hours)")
    ax.legend(loc="upper right", markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT / "anomaly_timeline.png", dpi=150)
    plt.close(fig)


def cutoff_tradeoff():
    r = json.loads(Path("data/processed/backtest_results.json").read_text())
    rows = pd.DataFrame(r["xgboost_cutoffs"]).sort_values("cutoff")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rows["cutoff"], rows["recall"] * 100, marker="o", label="High hours caught (%)")
    ax.plot(rows["cutoff"], rows["false_share"] * 100, marker="s", label="Alerts that were false (%)")
    ax.plot(rows["cutoff"], rows["share_hours_under_alert"] * 100, marker="^", label="Hours under alert (%)")
    ax.set_xlabel("Alert when forecast exceeds (µg/m³)")
    ax.set_ylabel("Percent")
    ax.set_title("Alert cut-off trade-off, 2014 test year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cutoff_tradeoff.png", dpi=150)
    plt.close(fig)


def forecast_comparison():
    d = pd.read_csv("data/processed/forecast_comparison.csv", parse_dates=["origin"])
    s, x = d[d["model"] == "sarima"].sort_values("origin"), d[d["model"] == "xgboost"].sort_values("origin")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s["origin"], s["actual"], color="black", marker="o", label="Actual PM2.5")
    ax.plot(s["origin"], s["predicted"], color="#d62728", marker="s", linestyle="--", label="SARIMA")
    ax.plot(x["origin"], x["predicted"], color="#1f77b4", marker="^", linestyle="--", label="XGBoost")
    ax.set_xlabel("Forecast origin (value predicted 24 hours later)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("24-hour-ahead forecasts, final 15 days of 2014")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "forecast_comparison_plot.png", dpi=150)
    plt.close(fig)


def model_comparison():
    """MAE of every hourly-origin forecast on the 2014 holdout, at 24 and 48 hours."""
    hz = json.loads(Path("data/processed/horizon_results.json").read_text())["horizons"]
    lstm_path = Path("data/processed/lstm_results.json")
    labels, values, colours = [], [], []
    palette = {"Persistence": "#b0b0b0", "Training mean": "#7f7f7f", "XGBoost": "#1f77b4", "LSTM": "#d62728"}
    for h in ("24h", "48h"):
        for name in ("Persistence", "Training mean", "XGBoost"):
            labels.append(f"{name}\n({h})"); values.append(hz[h]["models"][name]["mae"]); colours.append(palette[name])
        if h == "24h" and lstm_path.exists():
            lstm = json.loads(lstm_path.read_text())
            labels.append("LSTM\n(24h, mean of 3 seeds)"); values.append(lstm["lstm_mean"]["mae"]); colours.append(palette["LSTM"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, values, color=colours)
    ax.bar_label(bars, fmt="%.1f")
    ax.set_ylabel("Mean absolute error (µg/m³)")
    ax.set_title("Forecast error on the 2014 holdout, hourly origins (lower is better)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    anomaly_timeline()
    cutoff_tradeoff()
    forecast_comparison()
    model_comparison()
    print("Figures written to docs/")
