"""Offline evaluation for the demo. Writes plain CSV/JSON into demo/artifacts/.

Protocol (the flight is the unit of analysis everywhere):
  1. Hold out 20 % of flights (seed 42); they are not touched until the final check.
  2. Repeated 5-fold x 10 cross-validation on the remaining 80 %.
  3. Model comparison:
       - corrected resampled t-test (Nadeau & Bengio, 2003) on the CV folds,
       - paired per-flight errors on the hold-out set with a flight-level
         bootstrap 95 % CI for the difference in MAE / RMSE.
  4. Stress tests over all flights:
       - leave-one-configuration-out: predict a payload x speed x altitude
         combination that was never flown in training,
       - leave-one-date-out: predict every flight of a day from the other days.

Run from the project root:
    python demo/build_artifacts.py
"""
from __future__ import annotations

import json
import platform
from itertools import combinations

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, RepeatedKFold, train_test_split

from energy_models import (
    ARTIFACTS,
    FEATURES,
    MODEL_KEYS,
    MODEL_NAMES,
    SAMPLES_CSV,
    SEED,
    TARGET,
    load_flights,
    make_models,
)

N_SPLITS, N_REPEATS = 5, 10
HOLDOUT_FRAC = 0.2
N_BOOT = 10_000
LEVELS = [0.80, 0.90, 0.95]


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)),
        "MAPE": float(np.mean(np.abs((y - p) / y)) * 100),
    }


def corrected_resampled_ttest(diff, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003): variance inflated for overlapping training sets."""
    diff = np.asarray(diff, float)
    k = len(diff)
    se = np.sqrt((1.0 / k + n_test / n_train) * diff.var(ddof=1))
    t = diff.mean() / se
    return {"t": float(t), "df": k - 1, "p": float(2 * stats.t.sf(abs(t), k - 1))}


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    fl = load_flights()
    keys = [MODEL_KEYS[m] for m in MODEL_NAMES]

    # ---- 1. hold-out split -------------------------------------------------
    idx_dev, idx_hold = train_test_split(
        np.arange(len(fl)), test_size=HOLDOUT_FRAC, random_state=SEED
    )
    idx_dev, idx_hold = np.sort(idx_dev), np.sort(idx_hold)
    dev = fl.iloc[idx_dev].reset_index(drop=True)
    hold = fl.iloc[idx_hold].reset_index(drop=True)
    y_dev, y_hold = dev[TARGET].to_numpy(), hold[TARGET].to_numpy()
    (ARTIFACTS / "split_holdout.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "holdout_fraction": HOLDOUT_FRAC,
                "holdout_flights": [int(f) for f in hold.flight],
                "dev_flights": [int(f) for f in dev.flight],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---- 2. repeated CV on the development set ------------------------------
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    oof = {k: np.full((N_REPEATS, len(dev)), np.nan) for k in keys}
    fold_rows = []
    for i, (tr, te) in enumerate(rkf.split(dev)):
        rep = i // N_SPLITS
        for name, mdl in make_models().items():
            mdl.fit(dev.iloc[tr][FEATURES], y_dev[tr])
            pred = mdl.predict(dev.iloc[te][FEATURES])
            oof[MODEL_KEYS[name]][rep, te] = pred
            fold_rows.append(
                {"fold": i, "repeat": rep, "model": name, "n_train": len(tr),
                 "n_test": len(te), **metrics(y_dev[te], pred)}
            )
    folds = pd.DataFrame(fold_rows)
    folds.round(5).to_csv(ARTIFACTS / "cv_folds.csv", index=False)
    cv_summary = {
        name: {
            m: {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=1))}
            for m in ["MAE", "RMSE", "R2", "MAPE"]
        }
        for name, g in folds.groupby("model")
    }

    # ---- 3. hold-out evaluation ---------------------------------------------
    hold_pred = {}
    holdout_metrics = {}
    for name, mdl in make_models().items():
        mdl.fit(dev[FEATURES], y_dev)
        hold_pred[MODEL_KEYS[name]] = mdl.predict(hold[FEATURES])
        holdout_metrics[name] = metrics(y_hold, hold_pred[MODEL_KEYS[name]])

    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(hold), size=(N_BOOT, len(hold)))
    n_test = len(dev) // N_SPLITS
    n_train = len(dev) - n_test
    tests = []
    for a, b in combinations(MODEL_NAMES, 2):
        ka, kb = MODEL_KEYS[a], MODEL_KEYS[b]
        ea, eb = hold_pred[ka] - y_hold, hold_pred[kb] - y_hold
        boot = {
            "MAE": np.abs(ea)[boot_idx].mean(1) - np.abs(eb)[boot_idx].mean(1),
            "RMSE": np.sqrt((ea**2)[boot_idx].mean(1)) - np.sqrt((eb**2)[boot_idx].mean(1)),
        }
        point = {
            "MAE": float(np.abs(ea).mean() - np.abs(eb).mean()),
            "RMSE": float(np.sqrt((ea**2).mean()) - np.sqrt((eb**2).mean())),
        }
        fa = folds[folds.model == a].sort_values("fold")
        fb = folds[folds.model == b].sort_values("fold")
        for m in ["MAE", "RMSE"]:
            d = fa[m].to_numpy() - fb[m].to_numpy()
            tests.append(
                {
                    "model_a": a, "model_b": b, "metric": m,
                    # positive difference = model_b has the lower error
                    "cv_mean_diff": float(d.mean()),
                    **{f"cv_{k}": v for k, v in corrected_resampled_ttest(d, n_train, n_test).items()},
                    "holdout_diff": point[m],
                    "holdout_ci_lo": float(np.percentile(boot[m], 2.5)),
                    "holdout_ci_hi": float(np.percentile(boot[m], 97.5)),
                }
            )

    # ---- 4. leave-one-configuration-out over all flights ---------------------
    y_all = fl[TARGET].to_numpy()
    loco = {k: np.empty(len(fl)) for k in keys}
    for tr, te in LeaveOneGroupOut().split(fl, groups=fl.config):
        for name, mdl in make_models().items():
            mdl.fit(fl.iloc[tr][FEATURES], y_all[tr])
            loco[MODEL_KEYS[name]][te] = mdl.predict(fl.iloc[te][FEATURES])
    loco_metrics = {name: metrics(y_all, loco[MODEL_KEYS[name]]) for name in MODEL_NAMES}

    # ---- 4b. leave-one-date-out: are results inflated by same-day flights? ----
    lodo = {k: np.empty(len(fl)) for k in keys}
    for tr, te in LeaveOneGroupOut().split(fl, groups=fl.date):
        for name, mdl in make_models().items():
            mdl.fit(fl.iloc[tr][FEATURES], y_all[tr])
            lodo[MODEL_KEYS[name]][te] = mdl.predict(fl.iloc[te][FEATURES])
    lodo_metrics = {name: metrics(y_all, lodo[MODEL_KEYS[name]]) for name in MODEL_NAMES}

    # ---- 5. empirical prediction-interval half-widths ------------------------
    # "seen": the configuration has replicate flights in training (CV residuals)
    # "unseen": the configuration was never flown in training (LOCO residuals)
    intervals, coverage = {}, {}
    for name in MODEL_NAMES:
        k = MODEL_KEYS[name]
        res_cv = np.abs(oof[k] - y_dev[None, :]).ravel()
        res_loco = np.abs(loco[k] - y_all)
        intervals[name] = {
            "seen": {f"{lv:.2f}": float(np.quantile(res_cv, lv)) for lv in LEVELS},
            "unseen": {f"{lv:.2f}": float(np.quantile(res_loco, lv)) for lv in LEVELS},
        }
        coverage[name] = {
            f"{lv:.2f}": float(
                np.mean(np.abs(hold_pred[k] - y_hold) <= intervals[name]["seen"][f"{lv:.2f}"])
            )
            for lv in LEVELS
        }

    # ---- 6. OLS coefficients with 95 % CI (RQ1, raw units, all flights) -------
    X = np.column_stack([np.ones(len(fl)), fl[FEATURES].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, y_all, rcond=None)
    resid = y_all - X @ beta
    dof = len(fl) - X.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tcrit = stats.t.ppf(0.975, dof)
    lr_coef = [
        {
            "term": term, "coef": float(b), "se": float(s),
            "ci_lo": float(b - tcrit * s), "ci_hi": float(b + tcrit * s),
            "p": float(2 * stats.t.sf(abs(b / s), dof)),
        }
        for term, b, s in zip(["intercept"] + FEATURES, beta, se)
    ]

    # ---- 7. one out-of-sample prediction per flight (for error analysis) -----
    pf = fl[["flight", "config", "payload", "speed", "altitude", TARGET,
             "duration_s", "date", "local_time"]].copy()
    pf["split"] = "dev"
    pf.loc[idx_hold, "split"] = "holdout"
    for k in keys:
        pred = np.empty(len(fl))
        abs_err = np.empty(len(fl))
        pred[idx_dev] = oof[k].mean(axis=0)
        abs_err[idx_dev] = np.abs(oof[k] - y_dev[None, :]).mean(axis=0)
        pred[idx_hold] = hold_pred[k]
        abs_err[idx_hold] = np.abs(hold_pred[k] - y_hold)
        pf[f"pred_{k}"] = pred
        pf[f"abs_err_{k}"] = abs_err
        pf[f"loco_{k}"] = loco[k]
    pf.round(4).to_csv(ARTIFACTS / "per_flight.csv", index=False)

    # ---- 8. 1 Hz power profiles of the real flights ---------------------------
    n_profile_rows = 0
    if SAMPLES_CSV.exists():
        sm = pd.read_csv(SAMPLES_CSV, usecols=["flight", "time", "power_w", "rel_alt"])
        sm = sm[sm.groupby("flight").cumcount() % 5 == 0]
        sm.round({"time": 1, "power_w": 1, "rel_alt": 1}).to_csv(
            ARTIFACTS / "flight_profiles.csv", index=False
        )
        n_profile_rows = int(len(sm))

    summary = {
        "seed": SEED,
        "features": FEATURES,
        "target": TARGET,
        "protocol": {
            "holdout_fraction": HOLDOUT_FRAC, "cv_splits": N_SPLITS,
            "cv_repeats": N_REPEATS, "bootstrap_resamples": N_BOOT,
        },
        "data": {
            "n_flights": int(len(fl)), "n_dev": int(len(dev)), "n_holdout": int(len(hold)),
            "n_configs": int(fl.config.nunique()),
            "levels": {f: sorted(float(v) for v in fl[f].unique()) for f in FEATURES},
            "energy_min": float(y_all.min()), "energy_max": float(y_all.max()),
            "energy_mean": float(y_all.mean()),
            "profile_rows": n_profile_rows,
        },
        "cv": cv_summary,
        "holdout": holdout_metrics,
        "loco": loco_metrics,
        "leave_one_date_out": {"n_dates": int(fl.date.nunique()), "metrics": lodo_metrics},
        "tests": tests,
        "intervals": intervals,
        "coverage_holdout": coverage,
        "lr_coef": lr_coef,
        "env": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
        },
    }
    (ARTIFACTS / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"flights: {len(fl)} (dev {len(dev)}, hold-out {len(hold)}), configs: {fl.config.nunique()}")
    for name in MODEL_NAMES:
        c, h, lo = cv_summary[name], holdout_metrics[name], loco_metrics[name]
        print(
            f"{name:28s} CV MAE {c['MAE']['mean']:.3f}±{c['MAE']['std']:.3f} "
            f"RMSE {c['RMSE']['mean']:.3f} | hold-out MAE {h['MAE']:.3f} RMSE {h['RMSE']:.3f} "
            f"R2 {h['R2']:.3f} | LOCO MAE {lo['MAE']:.3f} RMSE {lo['RMSE']:.3f} "
            f"| by-date MAE {lodo_metrics[name]['MAE']:.3f} R2 {lodo_metrics[name]['R2']:.3f}"
        )
    print("flight dates:", fl.date.nunique())
    for t in tests:
        print(
            f"{t['model_a']} - {t['model_b']} [{t['metric']}]: CV diff {t['cv_mean_diff']:+.3f} "
            f"(p={t['cv_p']:.4f}); hold-out {t['holdout_diff']:+.3f} "
            f"[{t['holdout_ci_lo']:+.3f}, {t['holdout_ci_hi']:+.3f}]"
        )
    for name in MODEL_NAMES:
        print(name, "interval half-widths:", intervals[name], "coverage:", coverage[name])


if __name__ == "__main__":
    main()
