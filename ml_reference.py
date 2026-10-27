"""Reference forecasters for the ML table: a Poisson GLM and a linear
regression, validated exactly like the rf / xgb forecasters of
ml_demand_model.py (settings chosen by GroupKFold by ISO week inside
TUNE_SEASON, refit on it, tested once on TEST_SEASON's uncensored days against
the TUNE_SEASON month mean, with the same week-bootstrap CIs).

Reference rows only: they never enter sim_data.pkl, the forecasts the agents
see, the world or any RL result. Reads the same inputs as ml_demand_model.py
and writes only OUT_DIR, so every existing output stays as it is."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_season_env import SIM_DATA_PATH
from ml_demand_model import (load_raw, load_pms, load_bar, build_daily, add_availability,
                             week_groups, poisson_dev, nb_k, test_scores, deviance_gain,
                             make_model, FEATURES, SUMMER, TUNE_SEASON, TEST_SEASON, N_FOLDS,
                             MIN_TEST_DAYS)


# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR       = "ml_reference"
ML_COMPARISON = "ml_comparison.csv"        # rf / xgb rows: read, never written
ML_CV_GRID    = "ml_cv_grid.csv"

REF_NAMES = {"glm":    "Reference: Poisson GLM (L2)",
             "linear": "Reference: linear regression (Ridge)"}
# L2 penalty. PoissonRegressor: mean half-deviance + alpha/2 |w|^2, so the small
# end is about unpenalised (a natural limit). Ridge: squared error summed over days.
REF_GRID = {"glm":    [dict(alpha=a) for a in (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0)],
            "linear": [dict(alpha=a) for a in (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)]}
FLOOR = 0.01                                # a linear forecast can be negative: build_sim_data's floor

# Linear encoding: one-hot month and weekday, and the holiday flags (all 0/1, so
# the penalty treats them alike). Not season (constant in TUNE_SEASON) and not
# day_of_year / days_from_start: both seasons start on 3 Apr, so they are one
# linear trend, which a linear model would extrapolate; month covers the season.
MONTHS = list(range(int(SUMMER[0][:2]), int(SUMMER[1][:2]) + 1))
FLAGS  = ["easter_window", "whit_window", "public_holiday"]


def make_ref(name, cfg):
    encode = ColumnTransformer([
        ("calendar", OneHotEncoder(categories=[MONTHS, list(range(7))], sparse_output=False),
         ["month", "dow"]),
        ("flags", "passthrough", FLAGS)])
    if name == "glm":
        return make_pipeline(encode, PoissonRegressor(**cfg, solver="newton-cholesky", max_iter=1000))
    return make_pipeline(encode, Ridge(**cfg))


def predict_ref(name, model, X):
    p = model.predict(X)
    return np.maximum(p, FLOOR) if name == "linear" else p


# =============================================================================
# DATA (as ml_demand_model.py's main builds it)
# =============================================================================

def load_daily():
    """The daily table exactly as ml_demand_model.py's main builds it."""
    bk, _ = load_raw()
    pms = load_pms()
    bk = bk[bk["arrival"] <= pms["arrival"].max()].copy()   # offline stays known up to here
    bar_daily, bk["bar_ref"] = load_bar(bk)
    return add_availability(build_daily(bk, bar_daily), bk, pms)


def check_daily(daily, sim):
    """Fail unless the table is the one sim_data.pkl was built from: the same
    dates, online availability, hotel prices, requests and censored nights."""
    for season, s in sim["seasons"].items():
        d = daily[daily["season"] == season]
        same = (d.index.equals(s["dates"])
                and np.array_equal(d["online_avail"].values, s["online_avail"])
                and np.array_equal(d["hotel_price"].values, s["hotel_price"])
                and int(d["requests"].sum()) == s["actual"]["requests"]
                and int(d["censored"].sum()) == s["actual"]["sold_out_nights"])
        if not same:
            raise ValueError(f"Season {season}: the daily table differs from {SIM_DATA_PATH}'s "
                             "(did ml_demand_model.py's load filters change?)")


# =============================================================================
# VALIDATION (the protocol of ml_demand_model.train_models)
# =============================================================================

def cv_score(fit_predict, d):
    """choose_settings' cross-validation of one config: GroupKFold by ISO week
    inside TUNE_SEASON, each fold scored on its uncensored days.
    fit_predict(train rows, train y, test rows) -> predictions. (cv, se, oof)"""
    y, unc = d["requests"].values, ~d["censored"].values
    oof, scores = np.zeros(len(y)), []
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(d, y, week_groups(d.index)):
        oof[te] = fit_predict(d.iloc[tr], y[tr], d.iloc[te])
        scores.append(poisson_dev(y[te][unc[te]], oof[te][unc[te]]).mean())
    return np.mean(scores), np.std(scores, ddof=1) / np.sqrt(N_FOLDS), oof


def choose_ref(name, d):
    """Settings with the best mean CV deviance inside TUNE_SEASON (guardrail G1)."""
    if set(d["season"]) != {TUNE_SEASON}:
        raise RuntimeError(f"Settings must be chosen on {TUNE_SEASON} only, got {sorted(set(d['season']))}")
    rows = []
    for cfg in REF_GRID[name]:
        fp = lambda Xtr, ytr, Xte: predict_ref(name, make_ref(name, cfg).fit(Xtr, ytr), Xte)
        cv, se, oof = cv_score(fp, d)
        rows.append(dict(cfg=cfg, cv=cv, se=se, oof=oof))
    return min(rows, key=lambda r: r["cv"]), rows


def check_cv_protocol(d):
    """Fail unless cv_score reproduces ml_cv_grid.csv for xgb's chosen config
    (xgb refits are bit-identical): the same folds and the same scoring."""
    grid = pd.read_csv(ML_CV_GRID, float_precision="round_trip")
    row = grid[(grid["model"] == "xgb") & grid["chosen"]].iloc[0]
    cfg = dict(max_depth=int(row["max_depth"]), n_estimators=int(row["n_estimators"]))
    fp = lambda Xtr, ytr, Xte: make_model("xgb", cfg).fit(Xtr[FEATURES], ytr).predict(Xte[FEATURES])
    cv, se, _ = cv_score(fp, d)
    if (cv, se) != (row["cv_deviance"], row["cv_se"]):
        raise RuntimeError(f"CV protocol differs from ml_demand_model.py: xgb {cfg} gives "
                           f"{cv!r} ± {se!r}, {ML_CV_GRID} has {row['cv_deviance']!r} ± {row['cv_se']!r}")
    return cfg, cv


def evaluate(daily, out):
    y = daily["requests"].values
    tune = (daily["season"] == TUNE_SEASON).values
    test = (daily["season"] == TEST_SEASON).values
    if (tune & test).any() or not (tune | test).all():           # guardrail G1
        raise RuntimeError("TUNE_SEASON and TEST_SEASON must split the daily table")
    scored = test & ~daily["censored"].values
    if scored.sum() < MIN_TEST_DAYS:                              # guardrail G2
        raise RuntimeError(f"Only {scored.sum()} uncensored test days (< {MIN_TEST_DAYS})")
    groups = week_groups(daily.index[scored])
    month_mean = daily[tune].groupby("month")["requests"].mean()
    base = daily["month"].map(month_mean).values
    baseline = {"evaluation": "baseline", "model": f"Baseline: {TUNE_SEASON} month mean",
                **test_scores(y[scored], base[scored])}

    rows, grid_rows = [], []
    for name in REF_NAMES:
        chosen, cv_rows = choose_ref(name, daily[tune])
        cfg = chosen["cfg"]
        grid_rows += [{"model": name, **r["cfg"], "cv_deviance": r["cv"], "cv_se": r["se"],
                       "chosen": r is chosen} for r in cv_rows]
        unc = ~daily["censored"].values[tune]
        k = nb_k(y[tune][unc], chosen["oof"][unc])                # forecast error (reported)
        model = make_ref(name, cfg).fit(daily[tune], y[tune])
        p = predict_ref(name, model, daily[test])[scored[test]]
        gain, lo, hi = deviance_gain(y[scored], p, base[scored], groups)
        rows.append({"evaluation": f"test_{TEST_SEASON}", "model": REF_NAMES[name],
                     "config": str(cfg), "cv_deviance": chosen["cv"], "cv_se": chosen["se"],
                     **test_scores(y[scored], p, k),
                     "gain_vs_baseline": gain, "gain_lo": lo, "gain_hi": hi, "dispersion_k": k})
        out(f"  {REF_NAMES[name]:38s} {cfg}  CV dev {chosen['cv']:.3f} ± {chosen['se']:.3f}")
        if cfg["alpha"] == max(c["alpha"] for c in REF_GRID[name]):
            out(f"  !!! {REF_NAMES[name]}: chosen alpha at the large edge of REF_GRID")
        if name == "linear":
            n_oof = int((chosen["oof"] <= FLOOR).sum())
            n_test = int((model.predict(daily[test]) < FLOOR).sum())
            out(f"    linear forecasts floored at {FLOOR}: {n_oof} of {tune.sum()} out-of-fold "
                f"days, {n_test} of {test.sum()} test days")
    return rows, grid_rows, baseline, scored


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    log = []

    def out(*lines):
        for line in lines:
            print(line); log.append(line)

    daily = load_daily()
    check_daily(daily, joblib.load(SIM_DATA_PATH))
    tune = daily[daily["season"] == TUNE_SEASON]
    xgb_cfg, xgb_cv = check_cv_protocol(tune)
    out("=" * 72,
        f"  Reference forecasters: settings chosen inside {TUNE_SEASON} (GroupKFold by ISO week),",
        f"  tested once on the uncensored {TEST_SEASON} days. Not used by the simulator or the agents.",
        f"  Checks: daily table = {SIM_DATA_PATH}'s; CV protocol reproduces {ML_CV_GRID} "
        f"(xgb {xgb_cfg}: {xgb_cv:.6f})",
        "=" * 72)

    rows, grid_rows, baseline, scored = evaluate(daily, out)

    existing = pd.read_csv(ML_COMPARISON, float_precision="round_trip")
    ref_base = existing[existing["evaluation"] == "baseline"].iloc[0]
    if any(ref_base[c] != v for c, v in baseline.items()):
        raise RuntimeError(f"The baseline row differs from {ML_COMPARISON}'s: not the same test days")

    os.makedirs(OUT_DIR, exist_ok=True)
    ref = pd.DataFrame(rows)[[c for c in existing.columns if c in rows[0]]]
    ref.to_csv(os.path.join(OUT_DIR, "ml_reference_comparison.csv"), index=False)
    pd.DataFrame(grid_rows).to_csv(os.path.join(OUT_DIR, "ml_reference_cv_grid.csv"), index=False)
    table = pd.concat([existing, ref], ignore_index=True)
    table.to_csv(os.path.join(OUT_DIR, "ml_table.csv"), index=False)

    out("", f"  Test on {TEST_SEASON} ({scored.sum()} uncensored days); "
            "gain = baseline deviance - model deviance",
        f"  {'':46s}{'Dev.':>7}{'MAE':>7}{'Level':>7}{'Pattern':>9}{'Spearman':>10}"
        f"{'Cover80':>9}   Gain [95% CI]")
    for _, r in table.iterrows():
        label = r["model"] + (" (pooled CV, optimistic)" if r["evaluation"] == "pooled_cv_reference" else "")
        cover = f"{r['cover80']:9.2f}" if pd.notna(r["cover80"]) else " " * 9
        gain = (f"   {r['gain_vs_baseline']:+.3f} [{r['gain_lo']:+.3f}, {r['gain_hi']:+.3f}]"
                if pd.notna(r["gain_vs_baseline"]) else "")
        out(f"  {label:46s}{r['poisson_deviance']:7.3f}{r['MAE']:7.3f}{r['level']:7.2f}"
            f"{r['pattern_deviance']:9.3f}{r['spearman']:10.3f}{cover}{gain}")
        if r["evaluation"] == f"test_{TEST_SEASON}" and r["gain_hi"] < 0:
            out(f"  !!! {r['model']}: significantly worse than the baseline out of time")
    out("  The rf / xgb and baseline rows are ml_comparison.csv's, unchanged; the reference rows",
        f"  are in {OUT_DIR}/ml_reference_comparison.csv. Saved: {OUT_DIR}/")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
