"""common.py — shared loaders, feature sets, CV and metrics (seeded)."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, GroupKFold

SEED = 42
HERE = os.path.dirname(__file__)
P = os.path.join(HERE, "..", "data", "processed")
FIG = os.path.join(HERE, "..", "outputs", "figures")
TAB = os.path.join(HERE, "..", "outputs", "tables")

# Task A — flight-level energy (Wh)
A_FEATURES = ["payload", "speed", "altitude", "mean_wind", "max_wind", "std_wind", "v_start", "hour"]
A_TARGET = "energy_wh"
# Task B — sample-level instantaneous power (W)
B_FEATURES = ["payload", "speed", "altitude", "wind_speed", "wind_sin", "wind_cos",
              "ground_speed", "vertical_speed", "accel_mag", "ang_rate_mag", "rel_alt", "battery_voltage"]
B_TARGET = "power_w"


def load_flights():
    return pd.read_csv(os.path.join(P, "flights_full.csv"))


def load_samples(step=5):
    """Return samples decimated to ~1 Hz (every step-th row per flight)."""
    sm = pd.read_csv(os.path.join(P, "samples.csv"), low_memory=False)
    sm = sm[sm.groupby("flight").cumcount() % step == 0]
    sm = sm.reset_index(drop=True)
    rad = np.deg2rad(sm.wind_angle)
    sm["wind_sin"], sm["wind_cos"] = np.sin(rad), np.cos(rad)
    return sm


def metrics(y, yhat):
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mae = float(mean_absolute_error(y, yhat))
    r2 = float(r2_score(y, yhat))
    mape = float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y), 1e-6, None))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def cv_task_a(n_splits=5, n_repeats=10):
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)


def cv_task_b_splits(groups, n_splits=5, n_repeats=5):
    """Repeated GroupKFold (flight-wise): no sample of a flight leaks across folds."""
    out = []
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED + r)
        for tr, te in gkf.split(np.zeros(len(groups)), groups=groups):
            out.append((tr, te))
    return out
