"""06_repro_check.py — Reproducibility check (Lab M3).

Re-runs the Task A comparison twice with the fixed seed, compares the fold-level
metrics bit-for-bit, and records the software environment.
"""
import os
import sys
import json
import hashlib
import platform
import numpy as np
import pandas as pd
import sklearn, scipy, xgboost, matplotlib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
from common import SEED, TAB, A_FEATURES, A_TARGET, load_flights, metrics, cv_task_a

HERE = os.path.dirname(__file__)


def run_once():
    fl = load_flights(); X, y = fl[A_FEATURES].values, fl[A_TARGET].values
    rows = []
    for name, mdl in {
        "LR": make_pipeline(StandardScaler(), LinearRegression()),
        "Poly": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0)),
        "XGB": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                            random_state=SEED, n_jobs=1, verbosity=0),
    }.items():
        for i, (tr, te) in enumerate(cv_task_a(n_repeats=2).split(X)):
            mdl.fit(X[tr], y[tr]); rows.append({"model": name, "fold": i, **metrics(y[te], mdl.predict(X[te]))})
    return pd.DataFrame(rows)


r1, r2 = run_once(), run_once()
h1 = hashlib.md5(r1.round(10).to_csv(index=False).encode()).hexdigest()
h2 = hashlib.md5(r2.round(10).to_csv(index=False).encode()).hexdigest()
identical = bool(np.allclose(r1[["RMSE", "MAE", "R2"]].values, r2[["RMSE", "MAE", "R2"]].values, atol=1e-9))

# hash of processed data files
def fhash(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

P = os.path.join(HERE, "..", "data", "processed")
data_hashes = {f: fhash(os.path.join(P, f)) for f in ["flights_full.csv", "samples.csv", "flights_all.csv"]}

env = {
    "python": sys.version.split()[0], "platform": platform.platform(),
    "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
    "scipy": scipy.__version__, "xgboost": xgboost.__version__, "matplotlib": matplotlib.__version__,
    "seed": SEED,
}
report = {"run1_md5": h1, "run2_md5": h2, "identical_metrics": identical, "max_abs_diff": float(
    np.abs(r1[["RMSE", "MAE", "R2"]].values - r2[["RMSE", "MAE", "R2"]].values).max()),
    "data_md5": data_hashes, "environment": env}
with open(os.path.join(TAB, "reproducibility_report.json"), "w") as fh:
    json.dump(report, fh, indent=2)
with open(os.path.join(HERE, "..", "environment.txt"), "w") as fh:
    for k, v in env.items():
        fh.write(f"{k}=={v}\n" if k not in ("platform", "seed") else f"# {k}: {v}\n")
print(json.dumps(report, indent=2))
