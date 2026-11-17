"""01_build_dataset.py — Build sample-level and flight-level datasets.

Source: Rodrigues et al. (2021), "Data Collected with Package Delivery
Quadcopter Drone", CMU KiltHub, doi:10.1184/R1/12683453.v1
"""
import os, glob, json
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)
SEED = 42
np.random.seed(SEED)

params = pd.read_csv(os.path.join(RAW, "parameters.csv"))
frames = []
for f in sorted(glob.glob(os.path.join(RAW, "flights", "*.csv")), key=lambda p: int(os.path.basename(p)[:-4])):
    fid = int(os.path.basename(f)[:-4])
    d = pd.read_csv(f)
    d.insert(0, "flight", fid)
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df = df.merge(params, on="flight", how="left")
print("raw samples:", df.shape, "flights:", df.flight.nunique())

# ---------- cleaning log ----------
log = {}
log["raw_rows"] = int(len(df))
log["missing_total"] = int(df.isna().sum().sum())
dups = df.duplicated(subset=["flight", "time"]).sum()
log["duplicate_flight_time"] = int(dups)
df = df.drop_duplicates(subset=["flight", "time"])

# physical derived quantities
df["current_abs"] = df["battery_current"].abs()           # discharge current (A)
df["power_w"] = df["battery_voltage"] * df["current_abs"]  # instantaneous power (W)
df["ground_speed"] = np.sqrt(df.velocity_x**2 + df.velocity_y**2)
df["vertical_speed"] = df["velocity_z"]
df["accel_mag"] = np.sqrt(df.linear_acceleration_x**2 + df.linear_acceleration_y**2 + df.linear_acceleration_z**2)
df["ang_rate_mag"] = np.sqrt(df.angular_x**2 + df.angular_y**2 + df.angular_z**2)

# sanity filters (sensor glitches): voltage must be within LiPo 6S range, current < 200 A sensor limit
bad_v = (df.battery_voltage < 15) | (df.battery_voltage > 30)
bad_i = df.current_abs > 200
log["outlier_voltage_rows"] = int(bad_v.sum())
log["outlier_current_rows"] = int(bad_i.sum())
df = df[~(bad_v | bad_i)].copy()

# flight type: modeling set = standard route R1 with a numeric altitude and a payload
# level that has >1 replicate (750 g has a single flight). R2-R7 use different
# start points / variable altitude and are kept only in flights_all.csv.
df["altitude_num"] = pd.to_numeric(df.altitude, errors="coerce")
df["is_full_flight"] = (df.route == "R1") & df.altitude_num.notna() & (df.payload < 750)
log["flights_excluded_non_R1_or_750g"] = int(params[~((params.route == "R1") & (params.payload < 750))].shape[0])
df["hour"] = pd.to_datetime(df.local_time, format="%H:%M").dt.hour
df["date"] = pd.to_datetime(df.date)

# ---------- sample-level dataset (full flights only, 5 Hz) ----------
df["altitude"] = df["altitude_num"]
sample_cols = ["flight", "time", "route", "speed", "payload", "altitude", "hour",
               "wind_speed", "wind_angle", "battery_voltage", "current_abs", "power_w",
               "position_z", "ground_speed", "vertical_speed", "accel_mag", "ang_rate_mag"]
samples = df.loc[df.is_full_flight, sample_cols].reset_index(drop=True)
# relative altitude above take-off point
samples["rel_alt"] = samples.position_z - samples.groupby("flight").position_z.transform("first")

samples.to_csv(os.path.join(OUT, "samples.csv"), index=False)
log["sample_rows_full_flights"] = int(len(samples))
log["full_flights"] = int(samples.flight.nunique())

# ---------- flight-level dataset ----------
def agg(g):
    t = g.time.values
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0
    energy_wh = float(np.sum(g.power_w.values * dt) / 3600.0)
    charge_ah = float(np.sum(g.current_abs.values * dt) / 3600.0)
    return pd.Series({
        "duration_s": float(t[-1] - t[0]),
        "energy_wh": energy_wh,
        "charge_ah": charge_ah,
        "mean_power_w": float(g.power_w.mean()),
        "peak_power_w": float(g.power_w.max()),
        "mean_wind": float(g.wind_speed.mean()),
        "std_wind": float(g.wind_speed.std()),
        "max_wind": float(g.wind_speed.max()),
        "v_start": float(g.battery_voltage.iloc[0]),
        "v_end": float(g.battery_voltage.iloc[-1]),
        "v_drop": float(g.battery_voltage.iloc[0] - g.battery_voltage.iloc[-1]),
        "mean_ground_speed": float(g.ground_speed.mean()),
        "max_rel_alt": float((g.position_z - g.position_z.iloc[0]).max()),
        "n_samples": int(len(g)),
    })

fl = df.groupby("flight").apply(agg, include_groups=False).reset_index()
fl = fl.merge(params, on="flight", how="left")
fl["hour"] = pd.to_datetime(fl.local_time, format="%H:%M").dt.hour
fl["altitude"] = pd.to_numeric(fl.altitude, errors="coerce")
fl["is_full_flight"] = (fl.route == "R1") & fl.altitude.notna() & (fl.payload < 750)
fl["energy_per_s"] = fl.energy_wh / fl.duration_s
fl.to_csv(os.path.join(OUT, "flights_all.csv"), index=False)
full = fl[fl.is_full_flight].reset_index(drop=True)
full.to_csv(os.path.join(OUT, "flights_full.csv"), index=False)
log["flights_total"] = int(len(fl))
log["flights_full_route"] = int(len(full))
log["flights_ancillary_or_hover"] = int((~fl.is_full_flight).sum())
with open(os.path.join(OUT, "cleaning_log.json"), "w") as fh:
    json.dump(log, fh, indent=2)
print(json.dumps(log, indent=2))
print(full.describe().T)
