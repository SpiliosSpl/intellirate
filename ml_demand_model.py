import csv
import hashlib
import re
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import nbinom
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("XGBoost is required for ML2: pip install xgboost") from exc


# =============================================================================
# CONFIGURATION
# =============================================================================

XLSX_PATH = "BOOKINGS.xlsx"
ROOM_TYPE = "STD"
SEASONS   = [2025, 2026]
SUMMER    = ("04-01", "10-31")      # arrivals studied in every season (MM-DD)
# Physical STD rooms per season (room 108 left STD after 2025)
STD_ROOMS = {2025: ["101", "102", "103", "104", "108"] + [str(r) for r in range(301, 314)],
             2026: ["101", "102", "103", "104"] + [str(r) for r in range(301, 314)]}
CAPACITY  = {s: len(rooms) for s, rooms in STD_ROOMS.items()}

PII_COLUMNS = ["First Name", "Last Name", "Email", "Telephone", "Card",
               "Guest's Company", "Geo", "Location", "Region", "External ID"]

# Reconstructed Best Available Rate (output of bar_rates.py, in BAR_DIR)
BAR_DIR           = "bar_exports"
BAR_DAILY_PATH    = f"{BAR_DIR}/bar_daily.csv"     # double-occupancy BAR per stay night
BAR_BOOKINGS_PATH = f"{BAR_DIR}/bar_bookings.csv"  # BAR each booking was priced from
BAR_INPUTS_PATH   = f"{BAR_DIR}/bar_inputs.sha256" # hashes of the files bar_rates.py read

# Hotel PMS export: every STD stay, online and offline. Only arrival, departure,
# room and status are read, to find the rooms left for online sale each night.
PMS_PATH   = "hotel_pms_data.csv"
PMS_HEADER = ["Αρ.", "Άφιξη", "Αναχώρηση", "Δωμάτιο", "Τιμοκατάλογος", "Τιμή",
              "Κατάσταση", "A", "C", "Πηγή"]
PMS_STATUS = {"IN": True, "OUT": True}  # holds a room? (in house, checked out); others fail

# Fixed model parameters; the rest come from WORLD_PARAMS or GRID
RF_PARAMS = dict(n_estimators=500, random_state=42, n_jobs=-1)
# Poisson objective: the natural loss for daily counts. Small learning rate:
# only ~400 days of data.
XGB_PARAMS = dict(objective="count:poisson", learning_rate=0.03, min_child_weight=3,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

MODEL_NAMES = {"rf": "ML1 Random Forest", "xgb": "ML2 XGBoost (Poisson)"}

# The world (lambda_true) describes what happened: settings that reproduce the
# real seasons' shape and variance, fitted on every season. Not tuned on forecasts.
WORLD_PARAMS = {"rf":  dict(min_samples_leaf=3, max_features=0.6),
                "xgb": dict(max_depth=3, n_estimators=400)}

# The forecaster (lambda_forecast, what the agent sees) is validated out of
# time: settings chosen by GroupKFold inside TUNE_SEASON, then tested once on
# TEST_SEASON, which never chooses anything.
TUNE_SEASON, TEST_SEASON = 2025, 2026
GRID = {
    "rf":  [dict(min_samples_leaf=l, max_features=f)
            for l in (3, 5, 10, 20, 30, 50) for f in (0.4, 0.6, 0.8, 1.0)],
    "xgb": [dict(max_depth=d, n_estimators=n)
            for d in (1, 2, 3) for n in (50, 100, 200, 300, 400, 600, 800, 1000)],
}
NATURAL_LIMITS = {"max_features": 1.0, "max_depth": 1}   # grid ends that are not edges
N_FOLDS = 5                          # blocked by ISO week (no leakage in time)
N_BOOT, BOOT_SEED = 2000, 0          # test CIs: resample whole ISO weeks
MIN_TEST_DAYS = 100                  # uncensored test days required

# Realism checks on the world before calibration (flagged, not fatal)
WORLD_MIN_SPEARMAN = 0.5             # daily world vs actual requests
WORLD_MONTH_RANGE  = (0.8, 1.25)     # monthly world / actual requests

MIN_PROFILES_PER_MONTH = 20          # else fall back to the whole season

CALIB_SEASONS = 200                  # simulated seasons per calibration round
CALIB_SEED0   = 200_000              # never used by training or evaluation
CALIB_ROUNDS  = 20                   # per target, else the calibration fails
CALIB_TOL     = 0.005                # simulated vs real room-nights (G4)
WORLD_VAR_TOL = 0.01                 # simulated vs real daily variance of recorded requests (G3)


def make_model(name, cfg):
    if name == "rf":
        return RandomForestRegressor(**RF_PARAMS, **cfg)
    return XGBRegressor(**XGB_PARAMS, **cfg)


# =============================================================================
# 1. LOAD (as is)
# =============================================================================

def load_raw(path=XLSX_PATH):
    with warnings.catch_warnings():          # the export has no default cell style
        warnings.filterwarnings("ignore", "Workbook contains no default style", UserWarning)
        df = pd.read_excel(path, sheet_name=0)
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
    df = df[df["Status"].notna()].copy()                   # empty trailer row
    export_date = pd.to_datetime(df["Booking Date"]).max().normalize()

    df["arrival"]   = pd.to_datetime(df["Check-In"]).dt.normalize()
    df["departure"] = pd.to_datetime(df["Check-Out"]).dt.normalize()
    df["season"]    = df["arrival"].dt.year
    df = df[(df["Room Type"] == ROOM_TYPE) & (df["season"].isin(SEASONS))]
    md = df["arrival"].dt.strftime("%m-%d")
    df = df[(md >= SUMMER[0]) & (md <= SUMMER[1]) & (df["arrival"] <= export_date)].copy()

    df["nights"]       = (df["departure"] - df["arrival"]).dt.days
    df["rooms"]        = df["Rooms"].astype(int)
    df["room_nights"]  = df["Room-Nights"].astype(int)
    df["is_cancelled"] = df["Status"].eq("CL").astype(int)
    # Recorded rate: Total per room-night, as paid (channel, plan, occupancy
    # and promos included). bar_rates.py unwinds it to the BAR.
    df["price"]        = df["Total"] / df["room_nights"]
    return df, export_date


def file_hash(path):
    """SHA-256 of a file: bar_rates.py records its inputs, load_bar checks them."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def load_bar(bk):
    """From bar_rates.py: the daily BAR (Series by date) and each booking's
    reference BAR (aligned with bk). Fails loudly when the files are missing
    or out of date (an input changed since bar_rates.py ran)."""
    try:
        daily = pd.read_csv(BAR_DAILY_PATH, parse_dates=["date"]).set_index("date")["bar"]
        per_booking = pd.read_csv(BAR_BOOKINGS_PATH).set_index("ID")["bar_ref"]
        with open(BAR_INPUTS_PATH, encoding="utf-8") as fh:
            inputs = [line.split("  ", 1) for line in fh.read().splitlines()]
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc.filename} not found: run python bar_rates.py first") from exc
    changed = [path for digest, path in inputs if file_hash(path) != digest]
    if changed:
        raise ValueError(f"{', '.join(changed)} changed since bar_rates.py ran: "
                         "re-run python bar_rates.py")
    bar_ref = bk["ID"].astype("int64").map(per_booking)
    if bar_ref.isna().any():
        raise ValueError(f"{bar_ref.isna().sum()} bookings have no BAR: re-run python bar_rates.py "
                         "(BOOKINGS.xlsx or the rate files changed)")
    return daily, bar_ref.values


def load_pms(path=PMS_PATH):
    """Every STD stay in the hotel PMS, online and offline, whose status holds
    a room: arrival, departure and room (one room per row). Fails on a changed
    header, a malformed row, an unknown status or a room that is STD in no
    season, instead of guessing. Stays in a room that is STD in another season
    only (108 in 2026) are dropped. Some rows carry one stray empty field after
    the arrival date."""
    with open(path, encoding="cp1253", newline="") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    if not any(r[:len(PMS_HEADER)] == PMS_HEADER for r in rows):
        raise ValueError(f"{path}: header changed (expected {PMS_HEADER})")
    is_date = re.compile(r"\d{1,2}/\d{1,2}/\d{4}").fullmatch
    stays = [r[:2] + r[3:] if len(r) == len(PMS_HEADER) + 1 and r[2] == "" else r
             for r in rows if len(r) > 1 and is_date(r[1])]
    bad = sum(len(r) != len(PMS_HEADER) for r in stays)
    if bad:
        raise ValueError(f"{path}: {bad} stay rows do not have {len(PMS_HEADER)} fields")
    pms = pd.DataFrame(stays).iloc[:, [1, 2, 3, 6]]
    pms.columns = ["arrival", "departure", "room", "status"]
    for col in ("arrival", "departure"):
        pms[col] = pd.to_datetime(pms[col], format="%d/%m/%Y")
    if (pms["departure"] <= pms["arrival"]).any():
        raise ValueError(f"{path}: {(pms['departure'] <= pms['arrival']).sum()} stays "
                         "end on or before their arrival")
    unknown = set(pms["status"]) - set(PMS_STATUS)
    if unknown:
        raise ValueError(f"{path}: unknown status {sorted(unknown)}: add it to PMS_STATUS")
    pms = pms[pms["status"].map(PMS_STATUS)]
    never = set(pms["room"]) - set().union(*STD_ROOMS.values())
    if never:
        raise ValueError(f"{path}: rooms {sorted(never)} are STD in no season")
    std = np.array([room in STD_ROOMS.get(a.year, ()) for room, a in zip(pms["room"], pms["arrival"])])
    if not std.all():
        other = pms[~std]
        dropped = other.groupby([other["arrival"].dt.year, other["room"]]).size()
        print("  PMS: dropped stays in rooms not STD that season: "
              + ", ".join(f"{y} room {r} ({n})" for (y, r), n in dropped.items()))
    pms = pms[std].copy()
    pms["rooms"] = 1
    return pms[["arrival", "departure", "room", "rooms"]].reset_index(drop=True)


def nightly_rooms(stays):
    """Rooms occupied each night by stays (arrival, departure, rooms)."""
    stays = stays.reset_index(drop=True)
    s = stays.loc[stays.index.repeat((stays["departure"] - stays["arrival"]).dt.days)]
    night = s["arrival"] + pd.to_timedelta(s.groupby(level=0).cumcount(), unit="D")
    return s["rooms"].groupby(night.values).sum()


def occupied_rooms(pms, dates):
    """Rooms occupied in the PMS each night of dates. A room holds one stay
    per night, so a room-night entered twice counts once (flagged)."""
    s = pms.loc[pms.index.repeat((pms["departure"] - pms["arrival"]).dt.days)]
    night = s["arrival"] + pd.to_timedelta(s.groupby(level=0).cumcount(), unit="D")
    room_nights = pd.DataFrame({"night": night.values, "room": s["room"].values})
    twice = room_nights.duplicated()
    n_twice = (twice & room_nights["night"].isin(dates)).sum()
    if n_twice:
        print(f"  !!! {dates[0].year}: {n_twice} PMS room-nights held by two stays (counted once)")
    return room_nights[~twice].groupby("night").size().reindex(dates, fill_value=0)


def online_availability(bk, pms, dates, capacity):
    """Rooms left for online sale each night. Offline stays are the PMS stays
    not in the channel manager: capacity - (PMS rooms - confirmed online
    rooms), clipped to [0, capacity]. The PMS must cover every date (its first
    to its last arrival). Where it holds fewer rooms than the confirmed online
    stays (upgrades or room moves), no offline stay is left and the whole
    hotel opens online: flagged."""
    first, last = pms["arrival"].min(), pms["arrival"].max()
    if dates[0] < first or dates[-1] > last:
        raise ValueError(f"The PMS covers arrivals {first.date()} -> {last.date()}, "
                         f"not {dates[0].date()} -> {dates[-1].date()}")
    online = nightly_rooms(bk[bk["is_cancelled"] == 0]).reindex(dates, fill_value=0)
    offline = occupied_rooms(pms, dates) - online
    short = offline < 0
    if short.any():
        print(f"  !!! {dates[0].year}: the PMS holds fewer rooms than the confirmed online stays "
              f"on {short.sum()} nights ({-offline[short].sum()} room-nights): no offline stays "
              "there, so the whole hotel opens online")
    return (capacity - offline.clip(0, capacity)).to_numpy(int)


def add_availability(daily, bk, pms):
    """Rooms on sale online each night, and censored nights: sold out or
    closed online, so demand above the rooms on sale was never recorded.
    Censored nights are kept for training but never scored."""
    daily = daily.copy()
    daily["online_avail"] = np.concatenate([
        online_availability(bk, pms, daily.index[daily["season"] == s], CAPACITY[s])
        for s in SEASONS])
    sold = nightly_rooms(bk[bk["is_cancelled"] == 0]).reindex(daily.index, fill_value=0)
    daily["censored"] = sold.values >= daily["online_avail"].values
    return daily


# =============================================================================
# 2. DAILY TABLE + CALENDAR FEATURES
# =============================================================================

def orthodox_easter(year):
    """Meeus' Julian algorithm + 13 days (valid 1900–2099)."""
    a, b, c = year % 4, year % 7, year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    month = (d + e + 114) // 31
    day = (d + e + 114) % 31 + 1
    return pd.Timestamp(year, month, day) + pd.Timedelta(days=13)


def holiday_features(dates):
    out = pd.DataFrame(index=dates)
    easter = {y: orthodox_easter(y) for y in dates.year.unique()}
    e_off = np.array([(d - easter[d.year]).days for d in dates])
    w_off = e_off - 50                               # Whit Monday = Easter + 50
    out["easter_window"] = ((e_off >= -2) & (e_off <= 1)).astype(int)   # Fri..Mon
    out["whit_window"]   = ((w_off >= -2) & (w_off <= 0)).astype(int)   # Sat..Mon
    near = np.zeros(len(dates), dtype=int)
    for m, dd in [(3, 25), (5, 1), (8, 15), (10, 28)]:  # national holidays
        for y in dates.year.unique():
            near |= (np.abs((dates - pd.Timestamp(y, m, dd)).days) <= 1)
    out["public_holiday"] = near.astype(int)
    out["any_holiday"] = out.max(axis=1)
    return out


def hotel_price_per_date(bar_daily, dates):
    """The hotel's price on each date: the reconstructed double-occupancy BAR
    (bar_rates.py). Not smoothed: it is the price the hotel set. Any date the
    BAR file does not cover takes the nearest BAR."""
    return bar_daily.reindex(dates).interpolate().ffill().bfill()


FEATURES = ["season", "month", "day_of_year", "dow", "days_from_start",
            "easter_window", "whit_window", "public_holiday"]


def build_daily(bk, bar_daily):
    frames = []
    for season in SEASONS:
        s = bk[bk["season"] == season]
        start = s["arrival"].min()
        dates = pd.date_range(start, s["arrival"].max(), freq="D")   # load_raw caps at the export

        d = pd.DataFrame(index=dates)
        d["season"]    = season
        d["requests"]  = s.groupby("arrival")["rooms"].sum().reindex(dates, fill_value=0)
        d["confirmed"] = s[s["is_cancelled"] == 0].groupby("arrival")["rooms"].sum() \
                          .reindex(dates, fill_value=0)
        d["hotel_price"] = hotel_price_per_date(bar_daily, dates)
        d["month"] = dates.month
        d["day_of_year"] = dates.dayofyear
        d["dow"] = dates.dayofweek
        d["days_from_start"] = (dates - start).days
        d = d.join(holiday_features(dates))
        frames.append(d)
    daily = pd.concat(frames)
    daily.index.name = "date"
    return daily


# =============================================================================
# 3. ML MODELS: forecaster validated out of time, world fitted on every season
# =============================================================================

def week_groups(index):
    iso = index.isocalendar()
    return (iso.year.astype(str) + "-" + iso.week.astype(str)).values


def poisson_dev(y, p):
    """Poisson deviance of each day."""
    p = np.maximum(p, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / p), 0.0)
    return 2 * (t - (y - p))


def nb_k(y, p):
    """Negative-binomial k of the residuals around p (None: Poisson)."""
    resid_var, mean_lam = np.var(y - p), np.mean(p)
    return mean_lam ** 2 / (resid_var - mean_lam) if resid_var > mean_lam * 1.05 else None


def grid_edges(name, cfg):
    """Parameters of cfg at an end of GRID that is not a natural limit."""
    edges = []
    for key, v in cfg.items():
        values = [c[key] for c in GRID[name]]
        if v in (min(values), max(values)) and v != NATURAL_LIMITS.get(key):
            edges.append(key)
    return edges


def choose_settings(name, d):
    """Forecaster settings: GroupKFold by ISO week on TUNE_SEASON only, each
    fold scored on its uncensored days; the best mean deviance wins.
    Returns the chosen row ({cfg, cv, se}), its out-of-fold predictions and
    the table of every config."""
    if set(d["season"]) != {TUNE_SEASON}:                        # guardrail G1
        raise RuntimeError(f"Settings must be chosen on {TUNE_SEASON} only, got {sorted(set(d['season']))}")
    X, y, unc = d[FEATURES], d["requests"].values, ~d["censored"].values
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, week_groups(d.index)))
    rows, oofs = [], []
    for cfg in GRID[name]:
        oof, scores = np.zeros(len(y)), []
        for tr, te in folds:
            oof[te] = make_model(name, cfg).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            scores.append(poisson_dev(y[te][unc[te]], oof[te][unc[te]]).mean())
        rows.append(dict(cfg=cfg, cv=np.mean(scores), se=np.std(scores, ddof=1) / np.sqrt(N_FOLDS)))
        oofs.append(oof)
    best = min(range(len(rows)), key=lambda i: rows[i]["cv"])
    return rows[best], oofs[best], rows


def test_scores(y, p, k=None):
    """Forecast vs actual. 'level' is total forecast / total actual; the
    pattern deviance rescales the forecast to the actual total first (the
    agent sees the forecast divided by its maximum, i.e. only the pattern)."""
    p = np.maximum(p, 1e-6)
    out = {"poisson_deviance": poisson_dev(y, p).mean(),
           "MAE": np.mean(np.abs(y - p)), "RMSE": np.sqrt(np.mean((y - p) ** 2)),
           "level": p.sum() / y.sum(),
           "pattern_deviance": poisson_dev(y, p * y.sum() / p.sum()).mean(),
           "spearman": pd.Series(y).corr(pd.Series(p), method="spearman")}
    if k is not None:                       # 80% interval from the CV dispersion
        q = k / (k + p)
        out["cover80"] = np.mean((y >= nbinom.ppf(0.1, k, q)) & (y <= nbinom.ppf(0.9, k, q)))
    return out


def deviance_gain(y, p, base, groups):
    """Baseline deviance minus model deviance (positive: model better), with a
    95% CI from resampling whole ISO weeks. Every call draws the same weeks,
    so the CIs are paired between models and don't depend on the call order."""
    rng = np.random.default_rng(BOOT_SEED)
    diff = poisson_dev(y, base) - poisson_dev(y, p)
    weeks = [np.where(groups == w)[0] for w in np.unique(groups)]
    draws = [diff[np.concatenate([weeks[i] for i in rng.integers(len(weeks), size=len(weeks))])].mean()
             for _ in range(N_BOOT)]
    return diff.mean(), *np.percentile(draws, [2.5, 97.5])


def check_world(daily, name, fit):
    """Realism of the world before calibration: rescaled to each season's real
    total, it should follow the real days and months. Flagged, not fatal."""
    for season in SEASONS:
        d = daily[daily["season"] == season]
        m = (daily["season"] == season).values
        f = pd.Series(fit[m] * d["requests"].sum() / fit[m].sum(), index=d.index)
        rho = d["requests"].corr(f, method="spearman")
        month = f.groupby(d["month"]).sum() / d.groupby("month")["requests"].sum()
        flags = []
        if rho < WORLD_MIN_SPEARMAN:
            flags.append(f"daily Spearman {rho:.2f} < {WORLD_MIN_SPEARMAN}")
        out = month[(month < WORLD_MONTH_RANGE[0]) | (month > WORLD_MONTH_RANGE[1])]
        if len(out):
            flags.append("months outside " + str(WORLD_MONTH_RANGE) + ": "
                         + ", ".join(f"{mo}: {v:.2f}" for mo, v in out.items()))
        print(f"  World {name} {season}: daily Spearman {rho:.2f}, months world/actual "
              f"{month.min():.2f}-{month.max():.2f}" + ("" if flags else "  ok"))
        for fl in flags:
            print(f"  !!! World {name} {season}: {fl}")


def train_models(daily):
    X, y = daily[FEATURES], daily["requests"].values
    tune = (daily["season"] == TUNE_SEASON).values
    test = (daily["season"] == TEST_SEASON).values
    if (tune & test).any() or not (tune | test).all():           # guardrail G1
        raise RuntimeError("TUNE_SEASON and TEST_SEASON must split the daily table")
    scored = test & ~daily["censored"].values         # test days with full demand
    if scored.sum() < MIN_TEST_DAYS:                              # guardrail G2
        raise RuntimeError(f"Only {scored.sum()} uncensored test days (< {MIN_TEST_DAYS})")
    groups = week_groups(daily.index[scored])
    # Seasonal naive baseline: the TUNE_SEASON mean of the same month
    month_mean = daily[tune].groupby("month")["requests"].mean()
    base = daily["month"].map(month_mean).values

    print("=" * 72)
    print(f"  Forecaster: settings chosen inside {TUNE_SEASON} (GroupKFold by ISO week), "
          f"tested once on\n  {TEST_SEASON}: {scored.sum()} uncensored of {test.sum()} days "
          f"(censored: {(daily['censored'] & tune).sum()} in {TUNE_SEASON}, "
          f"{(daily['censored'] & test).sum()} in {TEST_SEASON})")
    print("=" * 72)
    results, table, grid_rows = {}, [], []
    for name in MODEL_NAMES:
        # 1. Forecaster settings: cross-validation inside TUNE_SEASON only
        chosen, oof_tune, rows = choose_settings(name, daily[tune])
        cfg = chosen["cfg"]
        grid_rows += [{"model": name, **r["cfg"], "cv_deviance": r["cv"], "cv_se": r["se"],
                       "chosen": r is chosen} for r in rows]
        unc = ~daily["censored"].values[tune]
        k = nb_k(y[tune][unc], oof_tune[unc])        # forecast error (reported)
        edges = grid_edges(name, cfg)
        print(f"  {MODEL_NAMES[name]:24s} {cfg}  CV dev {chosen['cv']:.3f} ± {chosen['se']:.3f}")
        if edges:
            print(f"  !!! {MODEL_NAMES[name]}: chosen {', '.join(edges)} at the edge of GRID")

        # 2. Test: refit on TUNE_SEASON, scored once on TEST_SEASON
        forecaster = make_model(name, cfg).fit(X[tune], y[tune])
        pred_test = forecaster.predict(X[test])
        p = pred_test[scored[test]]
        gain, lo, hi = deviance_gain(y[scored], p, base[scored], groups)
        table.append({"evaluation": f"test_{TEST_SEASON}", "model": MODEL_NAMES[name],
                      "config": str(cfg), "cv_deviance": chosen["cv"], "cv_se": chosen["se"],
                      **test_scores(y[scored], p, k),
                      "gain_vs_baseline": gain, "gain_lo": lo, "gain_hi": hi, "dispersion_k": k})

        # The agent's forecast is never in-sample: TUNE_SEASON out-of-fold,
        # TEST_SEASON from the model trained on TUNE_SEASON
        oof = np.full(len(y), np.nan)
        oof[tune], oof[test] = oof_tune, pred_test
        if np.isnan(oof).any():                                   # guardrail G1
            raise RuntimeError("Some days have no out-of-sample forecast")

        # Reference only: the pooled cross-validation over both seasons that was
        # used before. Neighbouring TEST_SEASON weeks are in training: optimistic.
        pooled = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, week_groups(daily.index)):
            pooled[te] = make_model(name, WORLD_PARAMS[name]).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        table.append({"evaluation": "pooled_cv_reference", "model": MODEL_NAMES[name],
                      "config": str(WORLD_PARAMS[name]), **test_scores(y[scored], pooled[scored]),
                      **dict(zip(("gain_vs_baseline", "gain_lo", "gain_hi"),
                                 deviance_gain(y[scored], pooled[scored], base[scored], groups)))})

        # 3. World: a description of the real seasons, fitted on all of them
        model = make_model(name, WORLD_PARAMS[name]).fit(X, y)
        fit = model.predict(X)
        # Shocks for the simulated world: the world's pattern is the in-sample
        # fit, which already holds part of the data's luck. Starting value of k
        # for calibrate_to_availability, which re-fits it: the world's total
        # variance equals the real one, on the fit after the per-season rescale
        # that build_sim_data applies: Var(y) = E[fit] + Var(fit) + E[fit^2] / k
        k_world = {}
        for season in SEASONS:                      # one k per season
            m = (daily["season"] == season).values
            f = fit[m] * y[m].sum() / fit[m].sum()
            extra = np.var(y[m]) - np.mean(f) - np.var(f)
            k_world[season] = np.mean(f ** 2) / extra if extra > 0.05 * np.mean(f) else None

        imp = pd.Series(model.feature_importances_, index=FEATURES)
        results[name] = dict(model=model, forecaster=forecaster, config=cfg, fit=fit, oof=oof,
                             k=k, k_world=k_world, importance=imp)
    table.append({"evaluation": "baseline", "model": f"Baseline: {TUNE_SEASON} month mean",
                  **test_scores(y[scored], base[scored])})

    print(f"\n  Test on {TEST_SEASON} (uncensored days); gain = baseline deviance - model deviance")
    print(f"  {'':46s}{'Dev.':>7}{'MAE':>7}{'Level':>7}{'Pattern':>9}{'Spearman':>10}"
          f"{'Cover80':>9}   Gain [95% CI]")
    for r in table:
        label = r["model"] + (" (pooled CV, optimistic)" if r["evaluation"] == "pooled_cv_reference" else "")
        cover = f"{r['cover80']:9.2f}" if "cover80" in r else " " * 9
        gain = (f"   {r['gain_vs_baseline']:+.3f} [{r['gain_lo']:+.3f}, {r['gain_hi']:+.3f}]"
                if "gain_vs_baseline" in r else "")
        print(f"  {label:46s}{r['poisson_deviance']:7.3f}{r['MAE']:7.3f}{r['level']:7.2f}"
              f"{r['pattern_deviance']:9.3f}{r['spearman']:10.3f}{cover}{gain}")
        if r["evaluation"] == f"test_{TEST_SEASON}" and r["gain_hi"] < 0:
            print(f"  !!! {r['model']}: significantly worse than the baseline out of time")
    print("  The pooled rows train on neighbouring test weeks: report them only as the optimistic bound.")
    pd.DataFrame(table).to_csv("ml_comparison.csv", index=False)
    pd.DataFrame(grid_rows).to_csv("ml_cv_grid.csv", index=False)
    print()
    for name, r in results.items():
        check_world(daily, name, r["fit"])
    print()
    imp = pd.DataFrame({MODEL_NAMES[n]: r["importance"] for n, r in results.items()})
    print("  Feature importance (world models):")
    print(imp.sort_values(MODEL_NAMES["rf"], ascending=False)
             .to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    return results


# =============================================================================
# 4. SIMULATION DATA
# =============================================================================

def build_sim_data(bk, daily, results):
    daily = daily.copy()
    for name, r in results.items():
        daily[f"true_{name}"] = np.maximum(r["fit"], 0.01)
        daily[f"fcst_{name}"] = np.maximum(r["oof"], 0.01)
        # Calibrate: each simulated season has exactly its real total demand
        for season in SEASONS:
            mask = daily["season"] == season
            daily.loc[mask, f"true_{name}"] *= (daily.loc[mask, "requests"].sum()
                                                / daily.loc[mask, f"true_{name}"].sum())

    seasons = {}
    for season in SEASONS:
        d = daily[daily["season"] == season]
        s = bk[bk["season"] == season].copy()
        # Each booking remembers the BAR it was priced from, so the simulator
        # can scale its real Total to any other BAR: the guest keeps their own
        # occupancy, channel and plan factors (Total = BAR x factors).
        s["orig_price"] = s["bar_ref"].values
        cols = ["rooms", "nights", "Total", "orig_price", "is_cancelled"]
        season_pool = s[cols].reset_index(drop=True)
        pools = {}
        for m in d["month"].unique():
            mp = s[s["arrival"].dt.month == m][cols].reset_index(drop=True)
            pools[int(m)] = mp if len(mp) >= MIN_PROFILES_PER_MONTH else season_pool

        ok = s[s["is_cancelled"] == 0]
        seasons[season] = {
            "capacity":        CAPACITY[season],
            "online_avail":    d["online_avail"].to_numpy(int),
            "dispersion_k":    {n: r["k_world"][season] for n, r in results.items()},
            "dates":           d.index,
            "lambda_true":     {n: d[f"true_{n}"].values for n in results},
            "lambda_forecast": {n: d[f"fcst_{n}"].values for n in results},
            "hotel_price":     d["hotel_price"].values,
            "p_ref":           float(np.median(d["hotel_price"])),
            "month":           d["month"].values,
            "dow":             d["dow"].values,
            "holiday":         d["any_holiday"].values,
            "profiles":        pools,
            "actual": {        # what really happened, for simulator validation
                "requests":    int(d["requests"].sum()),
                "requests_var": float(np.var(d["requests"])),   # daily, the G3 target
                "sold_out_nights": int(d["censored"].sum()),
                "confirmed":   int(len(ok)),          # bookings, as the env counts them
                "room_nights": int(ok["room_nights"].sum()),
                "revenue":     float(ok["Total"].sum()),
            },
        }
    return {"seasons": seasons, "models": MODEL_NAMES}


def simulate_hotel(sim, season, name):
    """The simulator at the hotel's prices on the calibration seasons, per
    season on average: room-nights sold, recorded requests (accepted, and
    cancelled whose stay fitted: what the booking data would hold) and their
    daily variance, and sold-out nights."""
    from hotel_season_env import HotelEnv      # the env reads sim_data: import here
    env = HotelEnv(season, name, sim=sim)
    rn, rec, var, full = 0, 0, 0.0, 0
    for i in range(CALIB_SEASONS):
        env.reset(seed=CALIB_SEED0 + i)
        day = np.empty(env.n_days)
        for t in range(env.n_days):
            info = env.step_price(env.hotel_price[t])[4]
            rn += info["room_nights"]
            day[t] = info["recorded"]
        rec, var = rec + day.sum(), var + day.var()
        full += (env.committed[:env.n_days] >= env.capacity).sum()
    n = CALIB_SEASONS
    return dict(room_nights=rn / n, recorded=rec / n, recorded_var=var / n, sold_out=full / n)


def calibrate_to_availability(sim):
    """Fit each world to the real season as the simulator records it: at the
    hotel's own prices and under the real online availability.
    - One demand factor, so that it sells the real room-nights (guardrail G4).
      Demand on nights closed online was never recorded, so the season-total
      scaling alone under-books.
    - dispersion_k, so that the daily variance of the recorded requests equals
      the real one (guardrail G3): the world the agents train in, compared like
      with like. If even pure Poisson is more volatile, k is None (flagged).
    At the hotel's price the price response cancels, so neither depends on the
    elasticity. Fails if a target is not met within CALIB_ROUNDS.
    Returns {(season, model): factor, k before and after, simulated vs real}."""
    out = {}
    for season, s in sim["seasons"].items():
        real = s["actual"]
        for name in s["lambda_true"]:
            k0 = s["dispersion_k"][name]
            theta = 1 / k0 if k0 else 0.0                 # 1/k; 0 is Poisson
            slope = np.mean(s["lambda_true"][name] ** 2) / real["requests_var"]   # first guess
            factor, last = 1.0, None
            for _ in range(CALIB_ROUNDS):                 # k: secant steps on theta (G3)
                s["dispersion_k"][name] = 1 / theta if theta > 0 else None
                for _ in range(CALIB_ROUNDS):             # demand factor at this k (G4)
                    r = simulate_hotel(sim, season, name)
                    ratio = real["room_nights"] / r["room_nights"]
                    if abs(ratio - 1) < CALIB_TOL:
                        break
                    s["lambda_true"][name] = s["lambda_true"][name] * ratio
                    factor *= ratio
                else:
                    raise RuntimeError(f"Calibration of {name} {season} did not converge "
                                       f"(simulated / real room-nights {1 / ratio:.3f})")
                gap = r["recorded_var"] / real["requests_var"] - 1
                floor = theta == 0 and gap > 0           # less variance is impossible
                if abs(gap) < WORLD_VAR_TOL or floor:
                    break
                if last and theta != last[0] and (gap - last[1]) / (theta - last[0]) > 0:
                    slope = (gap - last[1]) / (theta - last[0])
                last = (theta, gap)
                theta = max(theta - gap / slope, 0.0)
            else:
                raise RuntimeError(f"World {name} {season}: recorded-request variance "
                                   f"{1 + gap:.3f} x real after calibration")
            out[season, name] = dict(factor=factor, k0=k0, k=s["dispersion_k"][name],
                                     room_nights=1 / ratio - 1, var=1 + gap, floor=floor,
                                     recorded=r["recorded"] / real["requests"],
                                     sold_out=r["sold_out"])
    return out

# ==================plotting====================
def plot_fit(daily, results, path="demand_fit.png"):
    colours = {"rf": "#2E75B6", "xgb": "#27AE60"}
    fig, axes = plt.subplots(len(SEASONS), 1, figsize=(12, 3.8 * len(SEASONS)))
    for ax, season in zip(np.atleast_1d(axes), SEASONS):
        mask = (daily["season"] == season).values
        idx = daily.index[mask]
        actual = pd.Series(daily["requests"].values[mask], index=idx).resample("W").sum()
        ax.plot(actual.index, actual, "o-", color="#444", lw=1.2, ms=3, label="Actual")
        for n, r in results.items():
            fit = pd.Series(r["fit"][mask], index=idx).resample("W").sum()
            oof = pd.Series(r["oof"][mask], index=idx).resample("W").sum()
            ax.plot(fit.index, fit, color=colours[n], lw=2,
                    label=f"{MODEL_NAMES[n]} fit (simulator)")
            ax.plot(oof.index, oof, "--", color=colours[n], lw=1.2, alpha=0.8,
                    label=f"{MODEL_NAMES[n]} forecast ({TUNE_SEASON} out-of-fold, "
                          f"{TEST_SEASON} trained on {TUNE_SEASON})")
        ax.set_title(f"Weekly booking requests — STD, season {season}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    bk, export_date = load_raw()
    pms = load_pms()
    # Offline stays are known only up to the PMS's last arrival: the table ends there
    pms_end = pms["arrival"].max()
    n_late = (bk["arrival"] > pms_end).sum()
    bk = bk[bk["arrival"] <= pms_end].copy()
    bar_daily, bk["bar_ref"] = load_bar(bk)
    print("=" * 66)
    print(f"  Bookings: {len(bk)} STD, summer arrivals {SUMMER[0]} -> {SUMMER[1]} "
          f"(export date {export_date.date()},\n  PMS last arrival {pms_end.date()}: "
          f"{n_late} bookings arriving later excluded)")
    print("=" * 66 + "\n")

    daily = add_availability(build_daily(bk, bar_daily), bk, pms)
    for season in SEASONS:
        d = daily[daily["season"] == season]
        print(f"  Season {season}: {d.index.min().date()} -> {d.index.max().date()}  "
              f"({len(d)} days, {d['requests'].mean():.2f} requests/day, "
              f"BAR median EUR {d['hotel_price'].median():.0f}/night)")
    print()

    results = train_models(daily)
    sim = build_sim_data(bk, daily, results)
    calib = calibrate_to_availability(sim)
    print("  Online availability: capacity - (PMS rooms - confirmed online rooms)")
    for season, s in sim["seasons"].items():
        a = s["online_avail"]
        ok = bk[(bk["season"] == season) & (bk["is_cancelled"] == 0)]
        sold = nightly_rooms(ok).reindex(s["dates"], fill_value=0).values
        print(f"  Season {season}: {a.mean():.1f} of {s['capacity']} rooms online per night, "
              f"{(a == 0).sum()} nights closed, {(a <= 3).sum()} with <= 3 rooms, "
              f"{(s['capacity'] - a).sum()} offline room-nights; "
              f"real online sales above it on {(sold > a).sum()} nights")
        for name in s["lambda_true"]:
            c = calib[season, name]
            k0, k = ("Poisson" if v is None else f"{v:.2f}" for v in (c["k0"], c["k"]))
            print(f"    {name}: demand x {c['factor']:.3f}, k {k0} -> {k}; at the "
                  f"hotel's prices: room-nights {c['room_nights']:+.1%}, recorded requests "
                  f"{c['recorded']:.3f} x real, their daily variance {c['var']:.3f} x real, "
                  f"sold-out nights {c['sold_out']:.1f} (real {s['actual']['sold_out_nights']})")
            if c["floor"]:
                print(f"    !!! {name} {season}: even pure Poisson is more volatile than "
                      f"the real season (variance {c['var']:.3f} x real)")
    print()

    joblib.dump({"world": {n: r["model"] for n, r in results.items()},
                 "forecast": {n: r["forecaster"] for n, r in results.items()},
                 "configs": {n: {"world": WORLD_PARAMS[n], "forecast": r["config"]}
                             for n, r in results.items()},
                 "features": FEATURES}, "demand_models.pkl")
    joblib.dump(sim, "sim_data.pkl")
    plot_fit(daily, results)
    print("  Saved: demand_models.pkl, sim_data.pkl, ml_comparison.csv, ml_cv_grid.csv")
    print("\n  Next: python rl_agent.py")
