"""Shared definitions for the pre-flight energy demo.

Imported by build_artifacts.py (offline evaluation) and app.py (Streamlit demo),
so both always use the same inputs, models and data.

Prediction time t0 = just before take-off: only mission parameters that are
fixed in the flight plan are used as inputs. In-flight telemetry (measured wind,
velocity, duration, end voltage) is never an input.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent
FLIGHTS_CSV = PROJECT_DIR / "data" / "processed" / "flights_full.csv"
SAMPLES_CSV = PROJECT_DIR / "data" / "processed" / "samples.csv"
ARTIFACTS = DEMO_DIR / "artifacts"

FEATURES = ["payload", "speed", "altitude"]
TARGET = "energy_wh"

LR = "Linear Regression"
RF = "Random Forest"
POLY = "Polynomial Ridge (bậc 2)"
MODEL_NAMES = [LR, RF, POLY]
MODEL_KEYS = {LR: "lr", RF: "rf", POLY: "poly"}


def make_models() -> dict:
    """Fresh, unfitted models. Hyper-parameters match src/04_experiment.py."""
    return {
        LR: make_pipeline(StandardScaler(), LinearRegression()),
        RF: RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1
        ),
        POLY: make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0)
        ),
    }


def load_flights() -> pd.DataFrame:
    """One row per flight (route R1, 181 flights) from the project pipeline."""
    fl = pd.read_csv(FLIGHTS_CSV)
    fl["config"] = (
        fl.payload.astype(int).astype(str)
        + "g-"
        + fl.speed.astype(int).astype(str)
        + "mps-"
        + fl.altitude.astype(int).astype(str)
        + "m"
    )
    return fl


def fit_final_models(flights: pd.DataFrame) -> dict:
    """Models used for live prediction: trained on every available flight."""
    models = make_models()
    for mdl in models.values():
        mdl.fit(flights[FEATURES], flights[TARGET])
    return models
