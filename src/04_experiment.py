"""04_experiment.py — Model comparison with repeated CV + statistical tests (D3 / Lab M2)."""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import (SEED, TAB, FIG, A_FEATURES, A_TARGET, B_FEATURES, B_TARGET,
                    load_flights, load_samples, metrics, cv_task_a, cv_task_b_splits)

np.random.seed(SEED)
sns.set_theme(style="whitegrid")
PAL = ["#2563EB", "#F97316", "#10B981", "#8B5CF6"]


def models_A():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial Ridge (deg 2)": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9,
                                colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbosity=0),
    }


def models_B():
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Polynomial Ridge (deg 2)": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.6, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9,
                                colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbosity=0),
    }


rows = []
# ---- Task A ----
fl = load_flights()
XA, yA = fl[A_FEATURES].values, fl[A_TARGET].values
splitsA = list(cv_task_a().split(XA))
for name, mdl in models_A().items():
    for i, (tr, te) in enumerate(splitsA):
        mdl.fit(XA[tr], yA[tr])
        rows.append({"task": "A_energy", "model": name, "fold": i, **metrics(yA[te], mdl.predict(XA[te]))})
    print("done A", name)

# ---- Task B ----
sm = load_samples()
XB, yB, gB = sm[B_FEATURES].values, sm[B_TARGET].values, sm.flight.values
splitsB = cv_task_b_splits(gB)
for name, mdl in models_B().items():
    for i, (tr, te) in enumerate(splitsB):
        mdl.fit(XB[tr], yB[tr])
        rows.append({"task": "B_power", "model": name, "fold": i, **metrics(yB[te], mdl.predict(XB[te]))})
    print("done B", name)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(TAB, "experiment_cv_folds.csv"), index=False)


# ---- summary with 95% CI (t-distribution over folds) ----
def ci(x):
    x = np.asarray(x); n = len(x)
    se = x.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return x.mean(), x.mean() - h, x.mean() + h


summ = []
for (task, model), g in res.groupby(["task", "model"]):
    r = {"task": task, "model": model, "n_folds": len(g)}
    for m in ["RMSE", "MAE", "R2", "MAPE"]:
        mu, lo, hi = ci(g[m])
        r[f"{m}_mean"] = mu; r[f"{m}_ci_lo"] = lo; r[f"{m}_ci_hi"] = hi; r[f"{m}_std"] = g[m].std(ddof=1)
    summ.append(r)
summ = pd.DataFrame(summ).round(4)
summ.to_csv(os.path.join(TAB, "experiment_summary.csv"), index=False)


# ---- paired tests vs baseline (fold-wise), Holm correction ----
def holm(p):
    p = np.asarray(p, dtype=float); o = np.argsort(p); adj = np.empty_like(p); m = len(p); run = 0.0
    for k, idx in enumerate(o):
        run = max(run, (m - k) * p[idx]); adj[idx] = min(1.0, run)
    return adj


tests = []
for task, g in res.groupby("task"):
    base = g[g.model == "Linear Regression"].sort_values("fold").RMSE.values
    for model in [m for m in g.model.unique() if m != "Linear Regression"]:
        x = g[g.model == model].sort_values("fold").RMSE.values
        t, p_t = stats.ttest_rel(base, x)
        w, p_w = stats.wilcoxon(base, x)
        d = (base - x).mean() / (base - x).std(ddof=1)  # paired Cohen's d
        tests.append({"task": task, "comparison": f"Linear Regression vs {model}",
                      "mean_RMSE_base": base.mean(), "mean_RMSE_model": x.mean(),
                      "delta_RMSE": (base - x).mean(), "pct_improvement": 100 * (base - x).mean() / base.mean(),
                      "t_stat": t, "p_ttest": p_t, "wilcoxon_W": w, "p_wilcoxon": p_w, "cohen_d": d, "n_folds": len(x)})
tests = pd.DataFrame(tests)
tests["p_wilcoxon_holm"] = tests.groupby("task").p_wilcoxon.transform(holm)
tests["p_ttest_holm"] = tests.groupby("task").p_ttest.transform(holm)
tests = tests.round(6)
tests.to_csv(os.path.join(TAB, "experiment_stat_tests.csv"), index=False)

# pairwise best vs second-best
best_rows = []
for task, g in summ.groupby("task"):
    g2 = g.sort_values("RMSE_mean"); b1, b2 = g2.iloc[0].model, g2.iloc[1].model
    x1 = res[(res.task == task) & (res.model == b1)].sort_values("fold").RMSE.values
    x2 = res[(res.task == task) & (res.model == b2)].sort_values("fold").RMSE.values
    t, p_t = stats.ttest_rel(x2, x1); w, p_w = stats.wilcoxon(x2, x1)
    best_rows.append({"task": task, "best": b1, "second": b2, "delta_RMSE": (x2 - x1).mean(), "p_ttest": p_t, "p_wilcoxon": p_w})
best = pd.DataFrame(best_rows).round(6)
best.to_csv(os.path.join(TAB, "experiment_best_vs_second.csv"), index=False)

# ---- figure: fold RMSE distributions ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, (task, title, unit) in zip(ax, [("A_energy", "Task A — energy per flight", "RMSE (Wh)"),
                                       ("B_power", "Task B — instantaneous power", "RMSE (W)")]):
    g = res[res.task == task]
    order = summ[summ.task == task].sort_values("RMSE_mean").model.tolist()
    sns.boxplot(data=g, x="model", y="RMSE", order=order, ax=a, palette=PAL, hue="model", legend=False)
    sns.stripplot(data=g, x="model", y="RMSE", order=order, ax=a, color="k", size=2.5, alpha=.5)
    a.set_title(title); a.set_ylabel(unit); a.set_xlabel(""); a.tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig09_cv_rmse_boxplot.png"), dpi=200); plt.close()

print(summ[["task", "model", "RMSE_mean", "RMSE_ci_lo", "RMSE_ci_hi", "MAE_mean", "R2_mean", "MAPE_mean"]].to_string())
print(tests[["task", "comparison", "delta_RMSE", "pct_improvement", "p_ttest_holm", "p_wilcoxon_holm", "cohen_d"]].to_string())
print(best)
