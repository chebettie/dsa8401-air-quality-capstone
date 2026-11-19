"""
Shared evaluation helper: uncertainty on the difference in MAE between two
forecasts scored on the same origins.

Hourly forecast errors are strongly autocorrelated (a bad day is bad for
many consecutive hours), so the 8,000 or so holdout origins are far fewer
than 8,000 independent observations. An ordinary bootstrap over hours would
understate the uncertainty. This helper resamples whole calendar weeks
instead (a block bootstrap), which keeps the within-week dependence intact.

Assumption: errors more than a week apart are treated as independent. That
is reasonable for pollution episodes, which last days, but it is an
assumption and the interval should be read as approximate.
"""

import numpy as np
import pandas as pd


def block_bootstrap_mae_diff(index: pd.DatetimeIndex, abs_err_a: np.ndarray, abs_err_b: np.ndarray,
                             n_boot: int = 2000, seed: int = 42) -> dict:
    """MAE(a) minus MAE(b) with a 95% interval from a weekly block bootstrap.

    A positive value means forecast b has the lower error.
    """
    frame = pd.DataFrame({"a": abs_err_a, "b": abs_err_b}, index=index)
    weeks = frame.groupby(pd.Grouper(freq="7D")).agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
    weeks = weeks[weeks["n"] > 0]
    a, b, n = weeks["a"].values, weeks["b"].values, weeks["n"].values
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(weeks), len(weeks))
        diffs[i] = (a[pick].sum() - b[pick].sum()) / n[pick].sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mae_difference": float((a.sum() - b.sum()) / n.sum()),
            "ci95": [float(lo), float(hi)], "blocks": int(len(weeks)), "resamples": n_boot}
