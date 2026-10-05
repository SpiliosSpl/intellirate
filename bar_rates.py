import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_demand_model import load_raw, SEASONS, SUMMER


# =============================================================================
# CONFIGURATION
# =============================================================================

RULES_PATH     = "rate_rules.csv"
OCCUPANCY_PATH = "rate_occupancy.csv"
BOOKINGS_OUT   = "bar_bookings.csv"
DAILY_OUT      = "bar_daily.csv"
REPORT_OUT     = "bar_validation.txt"
PLOT_OUT       = "bar_daily.png"

CHILD_CHARGED_FROM = 4     # children aged >= 4 pay as adults
LADDER_STEP        = 0.5   # rates are set to the half euro
LADDER_TOL         = 0.05  # channels round net / discounted totals to the cent
LADDER_MIN_GROUPS  = 2     # a double level must be seen on >= 2 channel groups
PATH_TOL           = 0.03  # multi-night stay vs nightly BAR: max relative gap
PAIR_DAYS          = 3     # validation pairs: same night, booked <= 3 days apart
OUTLIER_DEV        = 0.15  # flag bookings > 15% away from the night's BAR


# =============================================================================
# 1. INPUT + RULES
# =============================================================================

def summer(season):
    return pd.Timestamp(f"{season}-{SUMMER[0]}"), pd.Timestamp(f"{season}-{SUMMER[1]}")


def prepare(df):
    df = df.copy()
    df["channel"] = np.where(df["Application"].eq("WEBHOTELIER"), "Direct", df["Source"])
    df["group"] = df["channel"].replace({"Expedia Hotel Collect": "Expedia"})
    df["bdate"] = pd.to_datetime(df["Booking Date"]).dt.normalize()
    ages = df["Channel Notes for Hotelier"].astype(str).str.extract(r"Ages?: ([\d,\s]+) years")[0]
    kids = df["Children"].fillna(0).astype(int)
    charged = ages.map(lambda a: sum(int(x) >= CHILD_CHARGED_FROM for x in a.split(","))
                       if isinstance(a, str) else np.nan)
    df["child_ages"] = ages
    df["adults_room"] = df["Adults"].fillna(0) / df["rooms"]
    df["kids_room"] = kids / df["rooms"]
    df["charged_kids_room"] = np.where(kids == 0, 0, charged / df["rooms"])   # NaN: age unknown
    return df


def load_rules(path=RULES_PATH):
    rules = pd.read_csv(path, dtype=str, keep_default_na=False)
    rules["factor"] = rules["factor"].astype(float)
    rules["round_to"] = pd.to_numeric(rules["round_to"].replace("", "0"))
    for c in ("book_from", "book_to"):
        rules[c] = pd.to_datetime(rules[c].replace("", None))
    return rules


def load_occupancy(path=OCCUPANCY_PATH):
    occ = pd.read_csv(path, dtype={"season": int, "occupancy": str, "status": str, "note": str})
    occ["occ_price"] = pd.to_numeric(occ["occ_price"])
    occ["double"] = pd.to_numeric(occ["double"])
    return occ


def rule_mask(df, r, use_dates=True):
    m = df["channel"].fillna("").str.fullmatch(r["channel"])
    if r["season"] != "*":
        m &= df["season"].eq(int(r["season"]))
    if r["field"] not in ("*", "@infer"):
        m &= df[r["field"]].astype(str).str.contains(r["pattern"], regex=True)
    if use_dates and pd.notna(r["book_from"]):
        m &= df["bdate"] >= r["book_from"]
    if use_dates and pd.notna(r["book_to"]):
        m &= df["bdate"] <= r["book_to"]
    return m


def infer_candidates(df, r, hits):
    """Bookings an @infer rule may apply to: its channel and season, not
    flagged by the rule named in its `pattern` (e.g. genius)."""
    return (rule_mask(df, r, use_dates=False) & ~hits[r["pattern"]]).values


def apply_rules(df, rules):
    df["factor"], df["rules"], df["round_to"] = 1.0, "", 0.0
    hits = {}
    for _, r in rules[rules["field"] != "@infer"].iterrows():
        m = rule_mask(df, r)
        hits[r["rule_id"]] = m
        df.loc[m, "factor"] *= r["factor"]
        df.loc[m, "rules"] += "+" + r["rule_id"]
        df.loc[m, "round_to"] = np.maximum(df.loc[m, "round_to"], r["round_to"])
    df["rules"] = df["rules"].str.lstrip("+").replace("", "flex")
    return hits


# =============================================================================
# 2. OCCUPANCY + 3. LADDERS
# =============================================================================

def occupancy_class(season, pax, occ):
    has_single = {s: occ[(occ["season"] == s) & (occ["occupancy"] == "single")]["occ_price"].notna().any()
                  for s in SEASONS}
    single = np.array([has_single.get(s, False) for s in season])
    return np.select([pax <= 1, pax == 2, pax >= 3],
                     [np.where(single, "single", "double"), "double", "triple"], "")


def snap(x):
    lvl = np.round(np.asarray(x, float) / LADDER_STEP) * LADDER_STEP
    return lvl, np.abs(np.asarray(x, float) - lvl) <= LADDER_TOL


def build_ladders(df, occ, exclude):
    """(season, occupancy) -> price levels. Double from data, others from the grid."""
    x = df[(df["nights"] == 1) & (df["occupancy"] == "double") & ~exclude].copy()
    x["lvl"], x["on"] = snap(x["occ_price"])
    n = x[x["on"]].groupby(["season", "lvl"])["group"].nunique()
    n = n[n >= LADDER_MIN_GROUPS]
    ladders = {(s, "double"): np.sort(n.loc[s].index.values) if s in n.index.get_level_values(0)
               else np.array([]) for s in SEASONS}
    for (s, o), g in occ.dropna(subset=["occ_price"]).groupby(["season", "occupancy"]):
        ladders[(s, o)] = np.sort(g["occ_price"].values)
    return ladders


def nearest_level(df, values, occupancy, ladders):
    """Nearest ladder level of each booking, and whether it is within tolerance:
    the channels' cent rounding plus, for rates rounded after the discounts
    (round_to), half the rounding step divided back through the discounts."""
    values, near = np.asarray(values, float), np.full(len(df), np.nan)
    for (s, o), lv in ladders.items():
        m = ((df["season"] == s).values & (np.asarray(occupancy) == o))
        if len(lv) and m.any():
            near[m] = lv[np.abs(values[m][:, None] - lv[None, :]).argmin(axis=1)]
    tol = LADDER_TOL + df["round_to"].values / 2 / df["factor"].values
    return near, np.abs(values - near) <= tol


def on_ladder(df, values, occupancy, ladders):
    return nearest_level(df, values, occupancy, ladders)[1]


def double_ratio(season, occupancy, occ):
    """Median double / occupancy price of the grid (1 for double, NaN if fixed)."""
    r = occ.assign(r=occ["double"] / occ["occ_price"]).groupby(["season", "occupancy"])["r"].median()
    return np.array([1.0 if o == "double" else r.get((s, o), np.nan)
                     for s, o in zip(season, occupancy)])


def infer_occupancy(df, occ, ladders, nightly=None):
    """Children of unknown age: one-night stays take the occupancy whose ladder
    fits; multi-night stays the one whose BAR matches the nightly BAR.
    Undecided: the child is charged."""
    df["occ_method"] = "known"
    unknown = df["charged_kids_room"].isna().values
    if not unknown.any():
        return
    lo = occupancy_class(df["season"], df["adults_room"], occ)
    hi = occupancy_class(df["season"], df["adults_room"] + df["kids_room"], occ)
    single = (df["nights"] == 1).values
    fit_lo = on_ladder(df, df["occ_price"], lo, ladders)
    fit_hi = on_ladder(df, df["occ_price"], hi, ladders)
    decided = single & (fit_lo != fit_hi)
    use_lo = decided & fit_lo
    if nightly is not None:
        expected = stay_mean(df, nightly)
        with np.errstate(invalid="ignore"):
            e_lo = np.abs(np.log(df["occ_price"].values * double_ratio(df["season"], lo, occ) / expected))
            e_hi = np.abs(np.log(df["occ_price"].values * double_ratio(df["season"], hi, occ) / expected))
        m = ~single & (np.fmin(e_lo, e_hi) <= PATH_TOL)
        decided |= m
        use_lo |= m & (np.nan_to_num(e_lo, nan=9) < np.nan_to_num(e_hi, nan=9))
    df.loc[unknown, "occupancy"] = np.where(use_lo, lo, hi)[unknown]
    df.loc[unknown, "occ_method"] = np.where(decided & single, "price",
                                             np.where(decided, "nightly", "prior"))[unknown]


# =============================================================================
# 4. INFERENCE + 5. BAR
# =============================================================================

def infer_rules(df, rules, hits, ladders, nightly=None):
    """@infer rules, per candidate booking. One-night stays: the ladder decides.
    Multi-night stays: the one-night-stay BAR of a first pass decides (double
    only). Undecided: the rule's booking-date window (the prior)."""
    df["infer"] = ""
    for _, r in rules[rules["field"] == "@infer"].iterrows():
        cand = infer_candidates(df, r, hits)
        plain, alt = df["occ_price"].values, df["occ_price"].values / r["factor"]
        decided = np.zeros(len(df), dtype=bool)
        choice = np.zeros(len(df), dtype=bool)

        single = (df["nights"] == 1).values
        fit_plain = on_ladder(df, plain, df["occupancy"], ladders)
        fit_alt = on_ladder(df, alt, df["occupancy"], ladders)
        m = single & (fit_plain != fit_alt)
        decided |= m
        choice[m] = fit_alt[m]

        if nightly is not None:
            expected = stay_mean(df, nightly)
            dbl = (df["occupancy"] == "double").values & ~single & ~np.isnan(expected)
            e_plain = np.abs(np.log(plain / expected))
            e_alt = np.abs(np.log(alt / expected))
            m = dbl & (np.minimum(e_plain, e_alt) <= PATH_TOL)
            decided |= m
            choice[m] = (e_alt < e_plain)[m]

        prior = rule_mask(df, r).values
        apply = cand & np.where(decided, choice, prior)
        method = np.where(decided & single, "ladder", np.where(decided, "nightly", "prior"))
        df.loc[cand, "infer"] = method[cand]
        df.loc[apply, "factor"] *= r["factor"]
        df.loc[apply, "rules"] = (df.loc[apply, "rules"].replace("flex", "")
                                  + "+" + r["rule_id"]).str.lstrip("+")
        hits[r["rule_id"]] = pd.Series(apply, index=df.index)
    df["occ_price"] = df["price"] / df["factor"]


def to_double(df, occ):
    """Occupancy price -> double BAR. Grid level for one-night stays on the
    grid; else the grid's median double/occupancy ratio; fixed prices: none."""
    df["bar"], df["bar_method"] = np.nan, ""
    d = df["occupancy"] == "double"
    df.loc[d, "bar"], df.loc[d, "bar_method"] = df.loc[d, "occ_price"], "direct"

    lvl, clean = snap(df["occ_price"])
    for (s, o), g in occ[occ["occupancy"] != "double"].groupby(["season", "occupancy"]):
        m = ((df["season"] == s) & (df["occupancy"] == o)).values
        if not m.any():
            continue
        grid = g.dropna(subset=["occ_price"]).set_index("occ_price")["double"]
        mapped = pd.Series(lvl[m]).map(grid).values
        on_grid = clean[m] & (df["nights"].values[m] == 1) & pd.Series(lvl[m]).isin(grid.index).values
        ratio = (g["double"] / g["occ_price"]).median()      # NaN when the grid has no doubles
        bar = np.where(on_grid, mapped, df["occ_price"].values[m] * ratio)
        method = np.where(on_grid & ~np.isnan(mapped), "grid",
                          np.where(np.isnan(bar), "fixed", "ratio"))
        df.loc[m, "bar"], df.loc[m, "bar_method"] = bar, method


def reconstruct(base, rules, occ, nightly=None):
    df = base.copy()
    hits = apply_rules(df, rules)
    df["occ_price"] = df["price"] / df["factor"]
    df["occupancy"] = occupancy_class(df["season"], df["adults_room"] + df["charged_kids_room"].fillna(0), occ)

    # First ladders only from bookings whose plan and occupancy are both known
    cand = np.zeros(len(df), dtype=bool)
    for _, r in rules[rules["field"] == "@infer"].iterrows():
        cand |= infer_candidates(df, r, hits)
    ladders = build_ladders(df, occ, exclude=cand | df["charged_kids_room"].isna().values)

    infer_occupancy(df, occ, ladders, nightly)
    infer_rules(df, rules, hits, ladders, nightly)
    ladders = build_ladders(df, occ, exclude=np.zeros(len(df), dtype=bool))
    near, df["on_ladder"] = nearest_level(df, df["occ_price"], df["occupancy"], ladders)
    fix = df["on_ladder"] & (df["nights"] == 1)            # drop the rounding of the rates
    df.loc[fix, "occ_price"] = near[fix.values]
    to_double(df, occ)
    return df, ladders, hits


# =============================================================================
# 6. NIGHTLY BAR
# =============================================================================

def wmedian(v, w):
    order = np.argsort(v)
    cw = np.cumsum(w[order])
    return v[order][np.searchsorted(cw, 0.5 * cw[-1])]


def nightly_estimate(obs):
    return obs.groupby("date").apply(lambda g: wmedian(g["bar"].values, g["w"].values))


def expand_nights(df):
    """One row per booked night (with the booking's BAR, if it has one yet)."""
    n = df["nights"].values
    cols = [c for c in ("ID", "season", "bar", "nights") if c in df.columns]
    rep = df.loc[df.index.repeat(n), cols].reset_index(drop=True)
    offset = np.concatenate([np.arange(k) for k in n])
    rep["date"] = df["arrival"].repeat(n).values + pd.to_timedelta(offset, unit="D")
    return rep


def stay_mean(df, nightly):
    """Mean nightly BAR over each booking's stay (NaN if a night is unknown)."""
    rep = expand_nights(df)
    rep["b"] = nightly.reindex(rep["date"]).values
    return rep.groupby("ID", sort=False)["b"].mean().reindex(df["ID"]).values


def single_night_bar(df):
    """Nightly BAR from one-night stays only, interpolated between them. The
    reference for multi-night stays, which never feed back into it."""
    nights = expand_nights(df[df["bar"].notna()])
    est = nightly_estimate(nights[nights["nights"] == 1].assign(w=1.0))
    return est.reindex(pd.date_range(nights["date"].min(), nights["date"].max())) \
              .interpolate().ffill().bfill()


def daily_bar(df):
    nights = expand_nights(df[df["bar"].notna()])
    single = nights[nights["nights"] == 1].assign(w=1.0)
    multi = nights[nights["nights"] > 1].copy()
    multi["w"] = 1.0 / multi["nights"]

    b = single_night_bar(df).reindex(multi["date"]).values
    # n-night Total = sum of its nightly rates: keep its sum, nightly shape
    multi["alloc"] = b * multi["bar"] * multi["nights"] / \
                     pd.Series(b, index=multi.index).groupby(multi["ID"]).transform("sum")
    obs = pd.concat([single, multi.assign(bar=multi["alloc"])[single.columns]])
    est = nightly_estimate(obs)

    # Summer window up to the last booked night (2026 stops at the export)
    last_night = expand_nights(df).groupby("season")["date"].max()
    frames = []
    for season in SEASONS:
        o = obs[obs["season"] == season]
        start, end = summer(season)
        dates = pd.date_range(start, min(last_night[season], end))
        g = o.groupby("date")
        d = pd.DataFrame(index=dates)
        d["season"] = season
        d["bar"] = est.reindex(dates)
        d["n_obs"] = g.size().reindex(dates, fill_value=0)
        d["n_single"] = single[single["season"] == season].groupby("date").size() \
                              .reindex(dates, fill_value=0)
        d["bar_min"] = g["bar"].min().reindex(dates)
        d["bar_max"] = g["bar"].max().reindex(dates)
        inside = (dates >= o["date"].min()) & (dates <= o["date"].max())
        d["source"] = np.where(d["bar"].notna(), "observed", np.where(inside, "interpolated", "filled"))
        d["bar"] = d["bar"].interpolate().ffill().bfill()
        frames.append(d)
    daily = pd.concat(frames)
    daily.index.name = "date"
    return daily.round(2)


# =============================================================================
# 7. VALIDATION
# =============================================================================

def recorded_price(df, dates):
    """The old hotel price: median recorded price per arrival, 7-day median."""
    daily = df.groupby("arrival")["price"].median().reindex(dates)
    return daily.rolling(7, center=True, min_periods=1).median().interpolate().ffill().bfill()


def close_pairs(df, col):
    """One-night bookings for the same night, booked <= PAIR_DAYS apart:
    share whose price agrees within 1% (BAR moves little in a few days)."""
    x = df[(df["nights"] == 1) & df[col].notna()][["ID", "arrival", "bdate", col, "season"]]
    p = x.merge(x, on=["arrival", "season"])
    p = p[(p["ID_x"] < p["ID_y"]) & ((p["bdate_x"] - p["bdate_y"]).dt.days.abs() <= PAIR_DAYS)]
    p["agree"] = (p[f"{col}_x"] / p[f"{col}_y"] - 1).abs() <= 0.01
    return p.groupby("season")["agree"].mean(), p.groupby("season").size()


def report(df, daily, ladders, rules, occ, hits):
    out = []
    w = out.append
    w("=" * 72)
    w("  BAR reconstruction — validation")
    w("=" * 72)

    w("\n1. Rule hits (bookings)")
    t = pd.DataFrame({rid: m.groupby(df["season"]).sum() for rid, m in hits.items()}).T
    t["status"] = rules.set_index("rule_id")["status"]
    w(t.to_string())

    w(f"\n2. Price ladders (double: seen on >= {LADDER_MIN_GROUPS} channel groups; others: rate_occupancy.csv)")
    for (s, o), lv in sorted(ladders.items()):
        w(f"  {s} {o:7s}: " + ", ".join(f"{v:g}" for v in lv))

    w("\n3. Occupancy (rooms booked) and how it was decided")
    w(pd.crosstab([df["season"], df["occupancy"]], df["occ_method"]).to_string())

    w("\n4. Occupancy grid check: one-night stays at each grid price, and the")
    w(f"   double quoted for the same night <= {PAIR_DAYS * 2} days apart")
    one = df[df["nights"] == 1]
    dbl = one[one["occupancy"] == "double"][["arrival", "bdate", "occ_price"]] \
        .rename(columns={"bdate": "db", "occ_price": "pair_double"})
    rows = []
    for _, g in occ.iterrows():
        if pd.isna(g["occ_price"]):
            continue
        b = one[(one["season"] == g["season"]) & (one["occupancy"] == g["occupancy"])
                & ((one["occ_price"] - g["occ_price"]).abs() <= LADDER_TOL)]
        p = b.merge(dbl, on="arrival")
        p = p[(p["bdate"] - p["db"]).dt.days.abs() <= PAIR_DAYS * 2]
        seen = p["pair_double"].round(1).value_counts().to_dict()
        rows.append(dict(season=g["season"], occupancy=g["occupancy"], occ_price=g["occ_price"],
                         grid_double=g["double"], bookings=len(b), pair_doubles=seen, status=g["status"]))
    w(pd.DataFrame(rows).to_string(index=False))

    w("\n5. Unflagged Booking.com bookings: inferred Genius?")
    c = df[df["infer"] != ""]
    if len(c):
        w(pd.crosstab([c["season"], c["infer"]], c["rules"].str.contains("genius_unflagged"))
          .rename(columns={True: "genius", False: "plain"}).to_string())

    w("\n6. One-night bookings on their ladder")
    w(one.groupby(["season", "occupancy", "channel"])["on_ladder"].agg(["size", "mean"])
         .rename(columns={"size": "n", "mean": "on_ladder"}).round(3).to_string())
    w("  total: " + ", ".join(f"{s} {one[one.season == s]['on_ladder'].mean():.1%}" for s in SEASONS))

    w(f"\n7. Same night, booked <= {PAIR_DAYS} days apart: share of pairs agreeing within 1%")
    for col, label in (("price", "recorded price"), ("occ_price", "occupancy price"), ("bar", "BAR (double)")):
        share, n = close_pairs(df, col)
        w(f"  {label:16s} " + ", ".join(f"{s} {share.get(s, np.nan):.1%} (n={n.get(s, 0)})" for s in SEASONS))

    w("\n8. Booking BAR / night's BAR, one-night stays (should be ~1 for every occupancy)")
    r = one.assign(ratio=one["bar"] / daily["bar"].reindex(one["arrival"]).values)
    w(r.groupby(["season", "occupancy", "bar_method"])["ratio"].agg(["size", "median"]).round(3).to_string())

    w("\n9. Daily BAR")
    for s in SEASONS:
        d = daily[daily["season"] == s]
        rec = recorded_price(df[df["season"] == s], d.index)
        w(f"  {s}: {d.index.min().date()} -> {d.index.max().date()}, {len(d)} nights, "
          f"{(d['source'] == 'observed').mean():.1%} observed ({(d['n_single'] > 0).mean():.1%} with a one-night stay)")
        w(f"        BAR median EUR {d['bar'].median():.2f} (min {d['bar'].min():.2f}, max {d['bar'].max():.2f}); "
          f"old recorded price median EUR {rec.median():.2f}; BAR / recorded median {(d['bar'] / rec).median():.3f}")
        w("        BAR levels (nights): " + ", ".join(f"{k:g}: {v}" for k, v in d["bar"].round(1).value_counts().head(8).items()))

    w(f"\n10. Outliers: one-night stays off their ladder, or > {OUTLIER_DEV:.0%} from the night's BAR")
    dev = (df["bar"] / daily["bar"].reindex(df["arrival"]).values - 1).abs()
    o = df[((df["nights"] == 1) & ~df["on_ladder"]) | (dev > OUTLIER_DEV)]
    w(o.assign(night_bar=daily["bar"].reindex(o["arrival"]).values)
       [["season", "bdate", "arrival", "nights", "channel", "occupancy", "rules", "price", "occ_price", "bar", "night_bar"]]
       .sort_values(["season", "arrival"]).to_string(float_format=lambda v: f"{v:.2f}"))
    return "\n".join(out)


def plot(df, daily, path=PLOT_OUT):
    fig, axes = plt.subplots(len(SEASONS), 1, figsize=(12, 3.8 * len(SEASONS)))
    for ax, s in zip(np.atleast_1d(axes), SEASONS):
        d = daily[daily["season"] == s]
        b = df[(df["season"] == s) & (df["nights"] == 1)]
        ax.scatter(b["arrival"], b["price"], s=8, color="#bbb", label="Recorded price (1-night stays)")
        ax.scatter(b["arrival"], b["bar"], s=8, color="#2E75B6", alpha=0.6, label="Implied BAR (1-night stays)")
        ax.plot(d.index, recorded_price(df[df["season"] == s], d.index), color="#888", lw=1.2,
                ls="--", label="Old hotel price (recorded, 7-day median)")
        ax.plot(d.index, d["bar"], color="#C0392B", lw=1.8, label="Daily BAR (double)")
        ax.set_title(f"Reconstructed BAR — STD, season {s}")
        ax.set_ylabel("EUR / night")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower center", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    bk, export_date = load_raw()
    base = prepare(bk)
    rules, occ = load_rules(), load_occupancy()

    # Pass 1 gives the one-night-stay BAR; pass 2 tests multi-night stays on it
    df, _, _ = reconstruct(base, rules, occ)
    df, ladders, hits = reconstruct(base, rules, occ, nightly=single_night_bar(df))
    daily = daily_bar(df)
    # BAR reference for every booking: its own BAR, else its nights' BAR
    df["bar_ref"] = df["bar"].fillna(pd.Series(stay_mean(df, daily["bar"]), index=df.index))

    cols = ["ID", "season", "bdate", "arrival", "nights", "channel", "Rate", "is_cancelled",
            "Adults", "Children", "child_ages", "occupancy", "occ_method", "price", "factor", "rules",
            "infer", "occ_price", "on_ladder", "bar", "bar_method", "bar_ref"]
    money = dict.fromkeys(["price", "factor", "occ_price", "bar", "bar_ref"], 4)
    df.assign(ID=df["ID"].astype("int64"))[cols].round(money).to_csv(BOOKINGS_OUT, index=False)
    daily.to_csv(DAILY_OUT)
    text = report(df, daily, ladders, rules, occ, hits)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    plot(df, daily)

    print(text)
    print(f"\n  Saved: {BOOKINGS_OUT}, {DAILY_OUT}, {REPORT_OUT}, {PLOT_OUT}")
    print("  Next: python ml_demand_model.py")
