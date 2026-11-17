"""03_baseline.py — Baseline models (D2 / Lab M1): mean predictor + Linear Regression."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import (SEED, TAB, FIG, A_FEATURES, A_TARGET, B_FEATURES, B_TARGET,
                    load_flights, load_samples, metrics, cv_task_a, cv_task_b_splits)

np.random.seed(SEED)
res_rows = []

# ---------- Task A ----------
fl = load_flights()
XA, yA = fl[A_FEATURES].values, fl[A_TARGET].values
for name, mdl in [("Mean (dummy)", DummyRegressor()),
                  ("Linear Regression", make_pipeline(StandardScaler(), LinearRegression()))]:
    for i, (tr, te) in enumerate(cv_task_a().split(XA)):
        mdl.fit(XA[tr], yA[tr])
        res_rows.append({"task": "A_energy", "model": name, "fold": i, **metrics(yA[te], mdl.predict(XA[te]))})

# coefficients of the full-data LR (interpretation)
lr = make_pipeline(StandardScaler(), LinearRegression()).fit(XA, yA)
coef = pd.DataFrame({"feature": A_FEATURES, "std_coef": lr[-1].coef_}).sort_values("std_coef", key=abs, ascending=False)
coef.to_csv(os.path.join(TAB, "baseline_lr_coefficients_taskA.csv"), index=False)
lr_raw = LinearRegression().fit(XA, yA)
pd.DataFrame({"feature": A_FEATURES, "coef": lr_raw.coef_}).assign(intercept=lr_raw.intercept_).to_csv(
    os.path.join(TAB, "baseline_lr_rawcoef_taskA.csv"), index=False)

pred = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()), XA, yA,
                         cv=KFold(5, shuffle=True, random_state=SEED))
plt.figure(figsize=(4.5, 4.5))
plt.scatter(yA, pred, s=18, alpha=.7, color="#2563EB")
lim = [yA.min() - 1, yA.max() + 1]
plt.plot(lim, lim, "k--", lw=1)
plt.xlabel("Actual energy (Wh)"); plt.ylabel("Predicted (Wh)")
plt.title("Baseline LR — Task A (5-fold CV predictions)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig08_baseline_taskA_pred.png"), dpi=200); plt.close()

# ---------- Task B ----------
sm = load_samples()
XB, yB, gB = sm[B_FEATURES].values, sm[B_TARGET].values, sm.flight.values
splits = cv_task_b_splits(gB)
for name, mdl in [("Mean (dummy)", DummyRegressor()),
                  ("Linear Regression", make_pipeline(StandardScaler(), LinearRegression()))]:
    for i, (tr, te) in enumerate(splits):
        mdl.fit(XB[tr], yB[tr])
        res_rows.append({"task": "B_power", "model": name, "fold": i, **metrics(yB[te], mdl.predict(XB[te]))})

res = pd.DataFrame(res_rows)
res.to_csv(os.path.join(TAB, "baseline_cv_folds.csv"), index=False)
summ = res.groupby(["task", "model"])[["RMSE", "MAE", "R2", "MAPE"]].agg(["mean", "std"]).round(3)
summ.to_csv(os.path.join(TAB, "baseline_summary.csv"))
print(summ)
print(coef)
print("samples used (1 Hz):", len(sm))
