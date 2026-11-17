"""02_eda.py — Exploratory Data Analysis (Lab M1 / D2)."""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(__file__)
P = os.path.join(HERE, "..", "data", "processed")
FIG = os.path.join(HERE, "..", "outputs", "figures")
TAB = os.path.join(HERE, "..", "outputs", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.0)
PAL = ["#2563EB", "#F97316", "#10B981", "#8B5CF6", "#EF4444", "#0EA5E9"]

fl = pd.read_csv(os.path.join(P, "flights_full.csv"))
sm = pd.read_csv(os.path.join(P, "samples.csv"), low_memory=False)
insights = {}

# ---- Table: descriptive statistics (flight level) ----
desc_cols = ["duration_s", "energy_wh", "mean_power_w", "peak_power_w", "mean_wind", "max_wind", "v_drop", "speed", "payload", "altitude"]
desc = fl[desc_cols].describe().T.round(2)
desc.to_csv(os.path.join(TAB, "eda_descriptive_flights.csv"))

# ---- Table: experimental design counts ----
design = fl.groupby(["payload", "speed", "altitude"]).size().reset_index(name="n_flights")
design.to_csv(os.path.join(TAB, "eda_design_counts.csv"), index=False)
fl.route.value_counts().rename_axis("route").reset_index(name="n").to_csv(os.path.join(TAB, "eda_route_counts.csv"), index=False)

# ---- Fig 1: distribution of energy per flight ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(fl.energy_wh, bins=25, kde=True, ax=ax[0], color=PAL[0])
ax[0].set_xlabel("Energy per flight (Wh)"); ax[0].set_title("Distribution of energy per flight")
sns.histplot(fl.duration_s, bins=25, kde=True, ax=ax[1], color=PAL[1])
ax[1].set_xlabel("Flight duration (s)"); ax[1].set_title("Distribution of flight duration")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig01_energy_duration_dist.png"), dpi=200); plt.close()

# ---- Fig 2: energy vs payload / speed / altitude ----
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
sns.boxplot(data=fl, x="payload", y="energy_wh", ax=ax[0], color=PAL[0]); ax[0].set_title("Energy vs payload (g)")
sns.boxplot(data=fl, x="speed", y="energy_wh", ax=ax[1], color=PAL[1]); ax[1].set_title("Energy vs cruise speed (m/s)")
sns.boxplot(data=fl, x="altitude", y="energy_wh", ax=ax[2], color=PAL[2]); ax[2].set_title("Energy vs altitude (m)")
for a in ax: a.set_ylabel("Energy (Wh)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig02_energy_vs_design.png"), dpi=200); plt.close()

# ---- Fig 3: mean power vs payload/speed (removes duration effect) ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.boxplot(data=fl, x="payload", y="mean_power_w", ax=ax[0], color=PAL[0]); ax[0].set_title("Mean power vs payload")
sns.boxplot(data=fl, x="speed", y="mean_power_w", ax=ax[1], color=PAL[1]); ax[1].set_title("Mean power vs cruise speed")
for a in ax: a.set_ylabel("Mean power (W)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig03_power_vs_design.png"), dpi=200); plt.close()

# ---- Fig 4: correlation heatmap ----
corr_cols = ["energy_wh", "duration_s", "mean_power_w", "peak_power_w", "payload", "speed", "altitude", "mean_wind", "max_wind", "v_start", "v_drop", "hour"]
corr = fl[corr_cols].corr(method="pearson")
corr.round(3).to_csv(os.path.join(TAB, "eda_correlation.csv"))
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
plt.title("Pearson correlation — flight-level variables"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig04_correlation.png"), dpi=200); plt.close()

# ---- Fig 5: power profile of one flight (phases) ----
ex = sm[sm.flight == int(fl.sort_values("energy_wh").iloc[len(fl)//2].flight)]
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
ax[0].plot(ex.time, ex.power_w, color=PAL[0]); ax[0].set_ylabel("Power (W)")
ax[1].plot(ex.time, ex.rel_alt, color=PAL[2]); ax[1].set_ylabel("Rel. altitude (m)")
ax[2].plot(ex.time, ex.ground_speed, color=PAL[1], label="ground speed"); ax[2].plot(ex.time, ex.wind_speed, color=PAL[3], alpha=.6, label="wind speed")
ax[2].set_ylabel("m/s"); ax[2].set_xlabel("Time (s)"); ax[2].legend()
fig.suptitle(f"Flight {int(ex.flight.iloc[0])} — payload {int(ex.payload.iloc[0])} g, speed {ex.speed.iloc[0]} m/s, altitude {int(ex.altitude.iloc[0])} m")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig05_flight_profile.png"), dpi=200); plt.close()

# ---- Fig 6: wind effect ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
sns.scatterplot(data=fl, x="mean_wind", y="mean_power_w", hue="payload", palette=PAL[:4], ax=ax[0]); ax[0].set_title("Mean power vs mean wind speed")
sns.scatterplot(data=fl, x="v_start", y="energy_wh", hue="payload", palette=PAL[:4], ax=ax[1]); ax[1].set_title("Energy vs start voltage (battery state)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig06_wind_voltage.png"), dpi=200); plt.close()

# ---- Fig 7: sample-level power distribution by vertical phase ----
sm["phase"] = np.where((sm.rel_alt < 2) & (sm.power_w < 60), "ground",
             np.where(sm.vertical_speed > 0.5, "climb",
             np.where(sm.vertical_speed < -0.5, "descent",
             np.where(sm.ground_speed > 1.0, "cruise", "hover"))))
phase_stats = sm.groupby("phase").power_w.agg(["count", "mean", "std", "median"]).round(1)
phase_stats.to_csv(os.path.join(TAB, "eda_phase_power.csv"))
plt.figure(figsize=(7, 3.8))
sns.boxplot(data=sm.sample(40000, random_state=42), x="phase", y="power_w", order=["ground", "climb", "cruise", "hover", "descent"], palette=PAL[:5])
plt.ylabel("Power (W)"); plt.title("Instantaneous power by flight phase (sample level)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig07_phase_power.png"), dpi=200); plt.close()

# ---- insights numbers ----
g = fl.groupby("payload").energy_wh.mean()
insights["energy_by_payload_wh"] = g.round(2).to_dict()
insights["energy_by_speed_wh"] = fl.groupby("speed").energy_wh.mean().round(2).to_dict()
insights["energy_by_altitude_wh"] = fl.groupby("altitude").energy_wh.mean().round(2).to_dict()
insights["power_by_payload_w"] = fl.groupby("payload").mean_power_w.mean().round(1).to_dict()
insights["power_by_speed_w"] = fl.groupby("speed").mean_power_w.mean().round(1).to_dict()
insights["corr_energy"] = corr["energy_wh"].drop("energy_wh").round(3).to_dict()
insights["corr_power"] = corr["mean_power_w"].drop("mean_power_w").round(3).to_dict()
insights["phase_power"] = phase_stats["mean"].to_dict()
insights["wind_power_corr"] = round(float(fl[["mean_wind", "mean_power_w"]].corr().iloc[0, 1]), 3)
insights["pct_energy_increase_500_vs_0"] = round(100 * (g[500] - g[0]) / g[0], 1)
with open(os.path.join(TAB, "eda_insights.json"), "w") as fh:
    json.dump(insights, fh, indent=2, default=float)
print(json.dumps(insights, indent=2, default=float))
print(design.to_string())
print(phase_stats)
