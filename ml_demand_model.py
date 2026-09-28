import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance

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
CAPACITY  = 17                      # physical STD rooms

PII_COLUMNS = ["First Name", "Last Name", "Email", "Telephone", "Card",
               "Guest's Company", "Geo", "Location", "Region", "External ID"]

PRICE_SMOOTHING_DAYS = 7            # rolling median of the daily price

RF_PARAMS = dict(n_estimators=500, min_samples_leaf=3, max_features=0.6,
                 random_state=42, n_jobs=-1)
# Poisson objective: the natural loss for daily counts. Shallow trees and a
# small learning rate: only ~390 days of data.
XGB_PARAMS = dict(objective="count:poisson", n_estimators=400, learning_rate=0.03,
                  max_depth=3, min_child_weight=3, subsample=0.8,
                  colsample_bytree=0.8, random_state=42, n_jobs=-1)

MODEL_NAMES = {"rf": "ML1 Random Forest", "xgb": "ML2 XGBoost (Poisson)"}


def make_model(name):
    return RandomForestRegressor(**RF_PARAMS) if name == "rf" else XGBRegressor(**XGB_PARAMS)
N_FOLDS = 5                          # blocked by ISO week (no leakage in time)
MIN_PROFILES_PER_MONTH = 20          # else fall back to the whole season


# =============================================================================
# 1. LOAD (as is)
# =============================================================================

def load_raw(path=XLSX_PATH):
    df = pd.read_excel(path, sheet_name=0)
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
    df = df[df["Status"].notna()].copy()                   # empty trailer row
    export_date = pd.to_datetime(df["Booking Date"]).max().normalize()

    df["arrival"]   = pd.to_datetime(df["Check-In"]).dt.normalize()
    df["departure"] = pd.to_datetime(df["Check-Out"]).dt.normalize()
    df["season"]    = df["arrival"].dt.year
    df = df[(df["Room Type"] == ROOM_TYPE) & (df["season"].isin(SEASONS))]
    df = df[df["arrival"] <= export_date].copy()

    df["nights"]       = (df["departure"] - df["arrival"]).dt.days
    df["rooms"]        = df["Rooms"].astype(int)
    df["room_nights"]  = df["Room-Nights"].astype(int)
    df["is_cancelled"] = df["Status"].eq("CL").astype(int)
    df["price"]        = df["Total"] / df["room_nights"]   # recorded, per night
    return df, export_date


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


def hotel_price_per_date(bookings, dates):
    """The hotel's price on each date. THIS VERSION: smoothed average
    recorded price per room-night. Later phase: reconstructed BAR."""
    daily = bookings.groupby("arrival")["price"].median().reindex(dates)
    smooth = daily.rolling(PRICE_SMOOTHING_DAYS, center=True, min_periods=1).median()
    return smooth.interpolate().ffill().bfill()


FEATURES = ["season", "month", "day_of_year", "dow", "days_from_start",
            "easter_window", "whit_window", "public_holiday"]


def build_daily(bk, export_date):
    frames = []
    for season in SEASONS:
        s = bk[bk["season"] == season]
        start = s["arrival"].min()
        end = min(s["arrival"].max(), export_date)
        dates = pd.date_range(start, end, freq="D")

        d = pd.DataFrame(index=dates)
        d["season"]    = season
        d["requests"]  = s.groupby("arrival")["rooms"].sum().reindex(dates, fill_value=0)
        d["confirmed"] = s[s["is_cancelled"] == 0].groupby("arrival")["rooms"].sum() \
                          .reindex(dates, fill_value=0)
        d["hotel_price"] = hotel_price_per_date(s, dates)
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
# 3. ML MODELS (same blocked cross-validation for both)
# =============================================================================

def train_models(daily):
    X, y = daily[FEATURES], daily["requests"].values
    iso = daily.index.isocalendar()
    groups = (iso.year.astype(str) + "-" + iso.week.astype(str)).values
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))

    # Naive baseline: season-month mean of the training folds
    oof_base = np.zeros(len(y))
    for tr, te in folds:
        means = daily.iloc[tr].groupby(["season", "month"])["requests"].mean()
        oof_base[te] = [means.get((s, mo), y[tr].mean())
                        for s, mo in zip(daily["season"].iloc[te], daily["month"].iloc[te])]

    results, rows = {}, []
    for name in MODEL_NAMES:
        oof = np.zeros(len(y))
        for tr, te in folds:
            oof[te] = make_model(name).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        model = make_model(name).fit(X, y)
        fit = model.predict(X)

        # Day-to-day variability around this model's forecast
        resid_var, mean_lam = np.var(y - oof), np.mean(oof)
        k = mean_lam ** 2 / (resid_var - mean_lam) if resid_var > mean_lam * 1.05 else None

        imp = pd.Series(model.feature_importances_, index=FEATURES)
        results[name] = dict(model=model, fit=fit, oof=oof, k=k, importance=imp)
        rows.append((MODEL_NAMES[name], oof, k))

    rows.append(("Baseline: season-month mean", oof_base, None))

    print("=" * 72)
    print("  ML comparison — out-of-fold (blocked by week) vs ACTUAL daily requests")
    print("=" * 72)
    print(f"  {'':30s}{'MAE':>8}{'RMSE':>8}{'Poisson dev.':>14}{'NB k':>8}")
    table = []
    for label, p, k in rows:
        mae = mean_absolute_error(y, p)
        rmse = mean_squared_error(y, p) ** 0.5
        dev = mean_poisson_deviance(y, np.maximum(p, 1e-6))
        print(f"  {label:30s}{mae:8.3f}{rmse:8.3f}{dev:14.3f}"
              f"{(f'{k:8.2f}' if k else '       -')}")
        table.append({"model": label, "MAE": mae, "RMSE": rmse,
                      "poisson_deviance": dev, "dispersion_k": k})
    pd.DataFrame(table).to_csv("ml_comparison.csv", index=False)
    print()
    imp = pd.DataFrame({MODEL_NAMES[n]: r["importance"] for n, r in results.items()})
    print("  Feature importance:")
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
        # Each booking remembers the hotel's price on its own arrival date,
        # so the simulator can scale its real Total to any other price.
        s["orig_price"] = d["hotel_price"].reindex(s["arrival"]).values
        cols = ["rooms", "nights", "Total", "orig_price", "is_cancelled"]
        season_pool = s[cols].reset_index(drop=True)
        pools = {}
        for m in d["month"].unique():
            mp = s[s["arrival"].dt.month == m][cols].reset_index(drop=True)
            pools[int(m)] = mp if len(mp) >= MIN_PROFILES_PER_MONTH else season_pool

        ok = s[s["is_cancelled"] == 0]
        seasons[season] = {
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
                "confirmed":   int(ok["rooms"].sum()),
                "room_nights": int(ok["room_nights"].sum()),
                "revenue":     float(ok["Total"].sum()),
            },
        }
    return {"seasons": seasons, "capacity": CAPACITY, "models": MODEL_NAMES,
            "dispersion_k": {n: r["k"] for n, r in results.items()}}

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
                    label=f"{MODEL_NAMES[n]} out-of-fold (forecast)")
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
    print("=" * 66)
    print(f"  Bookings: {len(bk)} STD (export date {export_date.date()}; "
          f"later stay dates excluded)")
    print("=" * 66 + "\n")

    daily = build_daily(bk, export_date)
    for season in SEASONS:
        d = daily[daily["season"] == season]
        print(f"  Season {season}: {d.index.min().date()} -> {d.index.max().date()}  "
              f"({len(d)} days, {d['requests'].mean():.2f} requests/day, "
              f"hotel price median EUR {d['hotel_price'].median():.0f}/night)")
    print()

    results = train_models(daily)
    sim = build_sim_data(bk, daily, results)

    joblib.dump({"models": {n: r["model"] for n, r in results.items()},
                 "features": FEATURES}, "demand_models.pkl")
    joblib.dump(sim, "sim_data.pkl")
    plot_fit(daily, results)
    print("  Saved: demand_models.pkl, sim_data.pkl, ml_comparison.csv")
    print("\n  Next: python rl_agent.py")
