"""
Feature-encoding helpers shared between the offline training pipeline
(src/data_pipeline.py) and the live API (api/main.py).

Keeping this logic in one place means the features the model is served
with at inference time are guaranteed to be built the same way as the
features it was trained on -- a common source of silent bugs in ML
systems when the two are implemented twice.
"""

import math

WIND_DIRECTIONS = ["NE", "NW", "SE", "cv"]


def cyclical_time_features(hour: int, month: int) -> dict:
    return {
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
    }


def wind_direction_one_hot(wind_dir: str) -> dict:
    wind_dir = wind_dir if wind_dir in WIND_DIRECTIONS else "cv"
    return {f"wind_dir_{d}": int(d == wind_dir) for d in WIND_DIRECTIONS}
