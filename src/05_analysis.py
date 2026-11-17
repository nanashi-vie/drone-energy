"""05_analysis.py — Error analysis, feature importance, clustering (Lab M2/M3, D3/D4)."""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import (SEED, TAB, FIG, A_FEATURES, A_TARGET, B_FEATURES, B_TARGET,
                    load_flights, load_samples, metrics)

np.random.seed(SEED)
sns.set_theme(style="whitegrid")
PAL = ["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#EF4444"]
out = {}

# =====================================================================
# Task A — best model (Polynomial Ridge) and XGBoost: CV predictions + residuals
# =====================================================================
fl = load_flights()
XA, yA = fl[A_FEATURES].values, fl[A_TARGET].values
kf = KFold(5, shuffle=True, random_state=SEED)
poly = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0))
xgbA = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                    random_state=SEED, n_jobs=-1, verbosity=0)
fl["pred_poly"] = cross_val_predict(poly, XA, yA, cv=kf)
fl["pred_xgb"] = cross_val_predict(xgbA, XA, yA, cv=kf)
fl["res_poly"] = fl[A_TARGET] - fl.pred_poly
fl["res_xgb"] = fl[A_TARGET] - fl.pred_xgb
fl.to_csv(os.path.join(TAB, "analysis_taskA_cv_predictions.csv"), index=False)

# residuals by design factor
err_design = fl.groupby(["payload", "speed", "altitude"]).agg(
    n=("flight", "size"), energy_mean=("energy_wh", "mean"),
    rmse_poly=("res_poly", lambda r: float(np.sqrt(np.mean(r ** 2)))),
    rmse_xgb=("res_xgb", lambda r: float(np.sqrt(np.mean(r ** 2))))).round(3).reset_index()
err_design.to_csv(os.path.join(TAB, "analysis_taskA_error_by_design.csv"), index=False)
for f in ["payload", "speed", "altitude"]:
    fl.groupby(f).agg(rmse_poly=("res_poly", lambda r: float(np.sqrt(np.mean(r ** 2)))),
                      rmse_xgb=("res_xgb", lambda r: float(np.sqrt(np.mean(r ** 2)))),
                      bias_poly=("res_poly", "mean"), bias_xgb=("res_xgb", "mean")).round(3).to_csv(
        os.path.join(TAB, f"analysis_taskA_error_by_{f}.csv"))

# worst flights
worst = fl.reindex(fl.res_poly.abs().sort_values(ascending=False).index)[
    ["flight", "payload", "speed", "altitude", "mean_wind", "max_wind", "v_start", "duration_s", "energy_wh", "pred_poly", "res_poly"]].head(10).round(2)
worst.to_csv(os.path.join(TAB, "analysis_taskA_worst_flights.csv"), index=False)

# normality of residuals + heteroscedasticity (Spearman |res| vs fitted)
out["taskA_shapiro_poly_p"] = float(stats.shapiro(fl.res_poly).pvalue)
out["taskA_spearman_absres_fitted_poly"] = [float(v) for v in stats.spearmanr(fl.res_poly.abs(), fl.pred_poly)]

# Fig 10: predicted vs actual + residual plots
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].scatter(fl.energy_wh, fl.pred_poly, s=18, alpha=.7, color=PAL[0], label="Polynomial Ridge")
ax[0].scatter(fl.energy_wh, fl.pred_xgb, s=18, alpha=.5, color=PAL[1], marker="^", label="XGBoost")
lim = [fl.energy_wh.min() - 1, fl.energy_wh.max() + 1]
ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlabel("Actual energy (Wh)"); ax[0].set_ylabel("Predicted (Wh)")
ax[0].set_title("Task A — CV predictions"); ax[0].legend()
ax[1].scatter(fl.pred_poly, fl.res_poly, s=18, alpha=.7, color=PAL[0]); ax[1].axhline(0, color="k", lw=1)
ax[1].set_xlabel("Fitted (Wh)"); ax[1].set_ylabel("Residual (Wh)"); ax[1].set_title("Residuals vs fitted (Poly Ridge)")
sns.boxplot(data=fl, x="speed", y="res_poly", ax=ax[2], color=PAL[2]); ax[2].axhline(0, color="k", lw=1)
ax[2].set_title("Residual by cruise speed (Poly Ridge)"); ax[2].set_ylabel("Residual (Wh)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig10_taskA_residuals.png"), dpi=200); plt.close()

# Fig 11: permutation importance (XGBoost, Task A) and standardized poly coefficients (top 10)
xgbA.fit(XA, yA)
pi = permutation_importance(xgbA, XA, yA, n_repeats=30, random_state=SEED, scoring="neg_root_mean_squared_error")
impA = pd.DataFrame({"feature": A_FEATURES, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance", ascending=False)
impA.round(4).to_csv(os.path.join(TAB, "analysis_taskA_perm_importance_xgb.csv"), index=False)
poly.fit(XA, yA)
names = poly[1].get_feature_names_out(A_FEATURES)
coefs = pd.DataFrame({"term": names, "coef": poly[-1].coef_}).sort_values("coef", key=abs, ascending=False)
coefs.round(4).to_csv(os.path.join(TAB, "analysis_taskA_poly_coefficients.csv"), index=False)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
sns.barplot(data=impA, x="importance", y="feature", ax=ax[0], color=PAL[0]); ax[0].set_title("Permutation importance — XGBoost (Task A)"); ax[0].set_xlabel("Increase in RMSE (Wh)")
top = coefs.head(10)
sns.barplot(data=top, x="coef", y="term", ax=ax[1], color=PAL[1]); ax[1].set_title("Top standardized terms — Polynomial Ridge (Task A)"); ax[1].set_xlabel("coefficient (Wh per SD)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig11_taskA_importance.png"), dpi=200); plt.close()

# =====================================================================
# Task B — XGBoost: hold-out by flight, per-phase error, importance, trace
# =====================================================================
sm = load_samples()
sm["phase"] = np.where((sm.rel_alt < 2) & (sm.power_w < 60), "ground",
             np.where(sm.vertical_speed > 0.5, "climb",
             np.where(sm.vertical_speed < -0.5, "descent",
             np.where(sm.ground_speed > 1.0, "cruise", "hover"))))
rng = np.random.RandomState(SEED)
flights = np.array(sorted(sm.flight.unique())); rng.shuffle(flights)
test_f = set(flights[: int(0.2 * len(flights))])
tr, te = ~sm.flight.isin(test_f), sm.flight.isin(test_f)
xgbB = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                    random_state=SEED, n_jobs=-1, verbosity=0)
xgbB.fit(sm.loc[tr, B_FEATURES].values, sm.loc[tr, B_TARGET].values)
sm.loc[te, "pred"] = xgbB.predict(sm.loc[te, B_FEATURES].values)
tst = sm[te].copy(); tst["res"] = tst.power_w - tst.pred
out["taskB_holdout_n_flights"] = int(len(test_f)); out["taskB_holdout_metrics"] = metrics(tst.power_w, tst.pred)
phase_err = tst.groupby("phase").agg(n=("res", "size"), mean_power=("power_w", "mean"),
                                     rmse=("res", lambda r: float(np.sqrt(np.mean(r ** 2)))), bias=("res", "mean")).round(2)
phase_err.to_csv(os.path.join(TAB, "analysis_taskB_error_by_phase.csv"))
pi = permutation_importance(xgbB, tst[B_FEATURES].values, tst.power_w.values, n_repeats=10, random_state=SEED, scoring="neg_root_mean_squared_error")
impB = pd.DataFrame({"feature": B_FEATURES, "importance": pi.importances_mean, "std": pi.importances_std}).sort_values("importance", ascending=False)
impB.round(3).to_csv(os.path.join(TAB, "analysis_taskB_perm_importance_xgb.csv"), index=False)

# energy from integrated power predictions (hold-out flights): does sample-level model recover flight energy?
def integrate(g, col):
    t = g.time.values; dt = np.diff(t, prepend=t[0]); dt[0] = 0
    return float(np.sum(g[col].values * dt) / 3600.0)
eint = tst.groupby("flight").apply(lambda g: pd.Series({"E_true": integrate(g, "power_w"), "E_pred": integrate(g, "pred")}), include_groups=False)
out["taskB_integrated_energy_metrics"] = metrics(eint.E_true.values, eint.E_pred.values)
eint.round(3).to_csv(os.path.join(TAB, "analysis_taskB_integrated_energy.csv"))

# Fig 12: trace of one hold-out flight + per-phase RMSE + importance
f0 = sorted(test_f)[len(test_f) // 2]
g = tst[tst.flight == f0]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(g.time, g.power_w, color="k", lw=1, label="actual"); ax[0].plot(g.time, g.pred, color=PAL[1], lw=1, label="XGBoost")
ax[0].set_title(f"Hold-out flight {f0}: power trace"); ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Power (W)"); ax[0].legend()
sns.barplot(data=phase_err.reset_index(), x="phase", y="rmse", ax=ax[1], color=PAL[0], order=["ground", "climb", "cruise", "hover", "descent"]); ax[1].set_title("RMSE by flight phase (hold-out)"); ax[1].set_ylabel("RMSE (W)")
sns.barplot(data=impB.head(8), x="importance", y="feature", ax=ax[2], color=PAL[2]); ax[2].set_title("Permutation importance — XGBoost (Task B)"); ax[2].set_xlabel("Increase in RMSE (W)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig12_taskB_analysis.png"), dpi=200); plt.close()

# =====================================================================
# Lab M2 — unsupervised: K-means vs DBSCAN on flight-state space (phase discovery)
# =====================================================================
feat = ["power_w", "ground_speed", "vertical_speed", "rel_alt"]
sub = sm.sample(12000, random_state=SEED)
Z = StandardScaler().fit_transform(sub[feat].values)
sil = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    sil[k] = float(silhouette_score(Z, km.labels_, sample_size=6000, random_state=SEED))
best_k = max(sil, key=sil.get)
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Z)
sub["kmeans"] = km.labels_
ari_km = float(adjusted_rand_score(sub.phase, sub.kmeans))
db = DBSCAN(eps=0.35, min_samples=30).fit(Z)
sub["dbscan"] = db.labels_
mask = sub.dbscan != -1
sil_db = float(silhouette_score(Z[mask], sub.dbscan[mask], sample_size=6000, random_state=SEED)) if sub.dbscan[mask].nunique() > 1 else float("nan")
ari_db = float(adjusted_rand_score(sub.phase[mask], sub.dbscan[mask]))
out["cluster_silhouette_by_k"] = sil; out["cluster_best_k"] = int(best_k)
out["cluster_kmeans_ari_vs_rule_phase"] = ari_km
out["cluster_dbscan_n_clusters"] = int(sub.dbscan[mask].nunique()); out["cluster_dbscan_noise_pct"] = float(100 * (~mask).mean())
out["cluster_dbscan_silhouette"] = sil_db; out["cluster_dbscan_ari_vs_rule_phase"] = ari_db
prof = sub.groupby("kmeans")[feat].mean().round(2); prof["n"] = sub.kmeans.value_counts().sort_index()
prof.to_csv(os.path.join(TAB, "analysis_cluster_profiles_kmeans.csv"))
pd.crosstab(sub.kmeans, sub.phase).to_csv(os.path.join(TAB, "analysis_cluster_vs_phase_crosstab.csv"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(list(sil.keys()), list(sil.values()), "o-", color=PAL[0]); ax[0].set_xlabel("k"); ax[0].set_ylabel("Silhouette"); ax[0].set_title("K-means: silhouette vs k")
sns.scatterplot(data=sub, x="ground_speed", y="power_w", hue="kmeans", palette="tab10", s=8, ax=ax[1], legend="brief"); ax[1].set_title(f"K-means (k={best_k}) in state space")
sns.scatterplot(data=sub, x="ground_speed", y="power_w", hue="phase", palette=PAL, s=8, ax=ax[2]); ax[2].set_title("Rule-based flight phases")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig13_clustering.png"), dpi=200); plt.close()

with open(os.path.join(TAB, "analysis_summary.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
print(impA); print(coefs.head(10)); print(phase_err); print(prof)
print(worst)
