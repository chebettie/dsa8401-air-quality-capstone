"""
Evaluates the Isolation Forest anomaly detector.

There is no ground-truth anomaly label in this dataset, so evaluation uses
four complementary checks and states what each can and cannot show:

  1. Episodes: consecutive flagged hours are grouped into events, so the
     flags can be inspected against dates and PM2.5 levels.
  2. Severity profile: what share of flagged hours are genuinely high, and
     what share of extreme hours the detector catches.
  3. Sensitivity: how the flagged set changes with the contamination
     setting (the 2% rate is a prior set in advance, not a finding).
  4. SYNTHETIC spike test: real hours are artificially altered to contain a
     sudden jump in PM2.5 and the detection rate is measured. These
     injected spikes are synthetic and are not findings about real events.
     The same test is repeated with an added "jump" feature
     (pm25 minus pm25_lag_1h) as an experiment; that variant is not deployed.
  5. REAL abrupt rises: hours in the actual data where PM2.5 rose sharply
     within one hour, and how many of them the detector flags. This is the
     real-data counterpart of the synthetic test.

Results are written to data/processed/anomaly_evaluation.json.

Run from the project root:
    python src/evaluate_anomaly_model.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from train_anomaly_model import FEATURE_COLUMNS, load_features

OUT = Path("data/processed/anomaly_evaluation.json")
RNG = np.random.default_rng(42)


def episodes(df: pd.DataFrame, max_gap_hours: int = 6) -> pd.DataFrame:
    flagged = df[df["anomaly_flag"] == 1].sort_values("timestamp")
    rows, start, prev = [], None, None
    for ts in flagged["timestamp"]:
        if start is None:
            start = prev = ts
        elif (ts - prev) <= pd.Timedelta(hours=max_gap_hours):
            prev = ts
        else:
            rows.append((start, prev)); start = prev = ts
    if start is not None:
        rows.append((start, prev))
    out = []
    for s, e in rows:
        seg = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)]
        fl = seg[seg["anomaly_flag"] == 1]
        out.append({"start": str(s), "end": str(e), "flagged_hours": int(len(fl)),
                    "peak_pm25": float(seg["pm25"].max()), "mean_pm25": float(fl["pm25"].mean())})
    return pd.DataFrame(out)


def main():
    df = load_features(Path("data/processed/pm25_features.csv"))
    bundle = joblib.load("models/isolation_forest.joblib")
    model = bundle["model"]
    X = df[FEATURE_COLUMNS]
    df["anomaly_flag"] = (model.predict(X.values) == -1).astype(int)

    res = {"rows": int(len(df)), "flagged": int(df["anomaly_flag"].sum())}

    # 1. Episodes
    ep = episodes(df)
    res["episode_count"] = int(len(ep))
    top = ep.sort_values("flagged_hours", ascending=False).head(8)
    res["top_episodes_by_flagged_hours"] = top.to_dict(orient="records")
    jan13 = ep[(pd.to_datetime(ep["start"]) >= "2013-01-10") & (pd.to_datetime(ep["start"]) <= "2013-01-14")]
    res["jan_2013_flagged_hours"] = int(jan13["flagged_hours"].sum())
    print(f"Flagged hours: {res['flagged']:,} in {len(ep)} episodes")
    print("Top episodes by flagged hours:\n", top.to_string(index=False))

    # 2. Severity profile
    fl, nf = df[df["anomaly_flag"] == 1], df[df["anomaly_flag"] == 0]
    res["median_pm25_flagged"] = float(fl["pm25"].median())
    res["median_pm25_unflagged"] = float(nf["pm25"].median())
    for t in (150, 250, 500):
        res[f"share_flagged_above_{t}"] = float((fl["pm25"] > t).mean())
        hi = df[df["pm25"] > t]
        res[f"hours_above_{t}"] = int(len(hi))
        res[f"share_of_hours_above_{t}_flagged"] = float(hi["anomaly_flag"].mean())
    res["flagged_by_year"] = {str(k): int(v) for k, v in fl.groupby(fl["timestamp"].dt.year).size().items()}
    print("\nSeverity:", {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()
                          if k.startswith("share") or k.startswith("median")})
    print("Flagged by year:", res["flagged_by_year"])

    # 3. Sensitivity to contamination
    base = set(fl.index)
    sens = []
    for c in (0.01, 0.02, 0.05):
        m = IsolationForest(n_estimators=200, contamination=c, random_state=42, n_jobs=-1).fit(X.values)
        f = set(df.index[m.predict(X.values) == -1])
        sens.append({"contamination": c, "flagged": len(f),
                     "share_above_250": float((df.loc[list(f), "pm25"] > 250).mean()),
                     "jaccard_with_2pct": len(f & base) / len(f | base)})
    res["sensitivity"] = sens
    print("\nSensitivity:", sens)

    # 4. SYNTHETIC spike injection
    pool = df[(df["anomaly_flag"] == 0) & (df["pm25"] < 100)]
    sample = pool.sample(n=2000, random_state=42)
    inj = {}
    for delta in (100, 200, 400):
        s = sample.copy()
        s["pm25"] = s["pm25_lag_1h"] + delta
        inj[delta] = float((model.predict(s[FEATURE_COLUMNS].values) == -1).mean())
    res["synthetic_spike_detection_deployed"] = inj
    print("\nSYNTHETIC spike detection rate (deployed model):", inj)

    # Experimental variant with an explicit jump feature (not deployed)
    def with_jump(frame):
        out = frame[FEATURE_COLUMNS].copy()
        out["pm25_jump_1h"] = frame["pm25"] - frame["pm25_lag_1h"]
        return out
    Xj = with_jump(df)
    mj = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1).fit(Xj.values)
    flj = df.index[mj.predict(Xj.values) == -1]
    inj_j = {}
    for delta in (100, 200, 400):
        s = sample.copy()
        s["pm25"] = s["pm25_lag_1h"] + delta
        inj_j[delta] = float((mj.predict(with_jump(s).values) == -1).mean())
    res["synthetic_spike_detection_with_jump_feature"] = inj_j
    res["jump_variant_overlap_with_deployed_flags"] = len(set(flj) & base) / len(set(flj) | base)
    res["jump_variant_share_flagged_above_250"] = float((df.loc[flj, "pm25"] > 250).mean())
    print("SYNTHETIC spike detection rate (jump-feature variant):", inj_j)
    print("Variant overlap (Jaccard) with deployed flags:", round(res["jump_variant_overlap_with_deployed_flags"], 3),
          "| share of its flags above 250:", round(res["jump_variant_share_flagged_above_250"], 3))

    # 5. REAL abrupt rises (genuine data, no synthetic alteration)
    df["jump_1h"] = df["pm25"] - df["pm25_lag_1h"]
    for thr in (100, 200):
        seg = df[df["jump_1h"] >= thr]
        res[f"real_rises_ge_{thr}"] = {"hours": int(len(seg)),
                                       "flagged": int(seg["anomaly_flag"].sum()),
                                       "share_flagged": float(seg["anomaly_flag"].mean())}
    top = df.nlargest(3, "pm25")[["timestamp", "pm25", "pm25_lag_1h", "anomaly_flag"]]
    res["top_readings"] = [{"timestamp": str(r.timestamp), "pm25": float(r.pm25),
                            "pm25_lag_1h": float(r.pm25_lag_1h), "flagged": int(r.anomaly_flag)}
                           for r in top.itertuples()]
    print("\nREAL abrupt rises:", {k: v for k, v in res.items() if k.startswith("real_rises")})
    print("Highest three readings:\n", top.to_string(index=False))

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
