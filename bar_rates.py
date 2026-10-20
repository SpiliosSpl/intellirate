import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import to_hex, to_rgba
from matplotlib.ticker import PercentFormatter
import zipfile
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.layout import Layout, ManualLayout
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.chart.text import RichText, Text
from openpyxl.chart.title import Title
from openpyxl.drawing.line import LineProperties
from openpyxl.drawing.spreadsheet_drawing import AbsoluteAnchor
from openpyxl.drawing.text import CharacterProperties, Font, Paragraph, ParagraphProperties, RegularTextRun
from openpyxl.drawing.xdr import XDRPoint2D, XDRPositiveSize2D

from ml_demand_model import load_raw, file_hash, SEASONS, SUMMER, XLSX_PATH, BAR_DIR


# =============================================================================
# CONFIGURATION
# =============================================================================

RULES_PATH     = "rate_rules.csv"
OCCUPANCY_PATH = "rate_occupancy.csv"
# Every output goes to BAR_DIR (bar_exports/)
BOOKINGS_OUT   = f"{BAR_DIR}/bar_bookings.csv"
DAILY_OUT      = f"{BAR_DIR}/bar_daily.csv"
REPORT_OUT     = f"{BAR_DIR}/bar_validation.txt"
PLOT_OUT       = f"{BAR_DIR}/bar_daily.png"
PLOT_PCT_OUT   = f"{BAR_DIR}/bar_daily_pct.png"
INPUTS_OUT     = f"{BAR_DIR}/bar_inputs.sha256"   # hashes of the inputs, checked by ml_demand_model.load_bar
EXCEL_FONT     = "Verdana" # Excel copies of the plots: nearest Windows font to matplotlib's DejaVu Sans

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
        rules[c] = pd.to_datetime(rules[c].replace("", np.nan))
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
    if (df["occupancy"] == "").any():          # else a silent NaN BAR
        raise ValueError(f"{(df['occupancy'] == '').sum()} bookings have no occupancy class "
                         "(people per room is not a whole number)")
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


def p_ref(df, daily, season):
    """The simulator's reference price (multiplier 1.00): the median daily BAR
    over the season's arrival dates, as in ml_demand_model.build_sim_data."""
    a = df.loc[df["season"] == season, "arrival"]
    return daily.loc[a.min():a.max(), "bar"].median()


def excel_plot(fig, panels, path):
    """The figure as editable Excel charts: sheet 'charts' has one chart per
    season, drawn from that season's data sheet. Size, plot area, axis ranges,
    ticks, title, legend, colours, transparency, markers and lines are copied
    from the matplotlib figure. Excel ticks start at the axis minimum, so each
    axis starts at the last tick before the data instead of matplotlib's margin."""
    emu_pt, emu_in = 12700, 914400
    offset = mdates.date2num(pd.Timestamp("1899-12-30"))   # matplotlib -> Excel date
    inv = fig.transFigure.inverted()
    renderer = fig.canvas.get_renderer()
    n = len(panels)
    alphas = {}                                            # colour -> alpha, set in the XML below
    # Excel keeps half the widest date label inside the chart (matplotlib lets
    # it reach the margin): widen each chart on the right only, so the plot area
    # keeps matplotlib's size and position
    w, h = fig.get_size_inches()
    label = max(t.get_window_extent(renderer).width for p in panels for t in p[0].get_xticklabels()) / fig.dpi
    right = w * (1 - max(p[0].get_position().x1 for p in panels))
    cw = w + max(0.0, 1.25 * label / 2 + 5 / 72 - right)   # chart width, inches
    sx = w / cw

    def colour(c, alpha=None):
        rgba = to_rgba(c, alpha)
        hexc = to_hex(rgba[:3]).lstrip("#").upper()
        if rgba[3] < 1:
            alphas[hexc] = rgba[3]
        return hexc

    def font(size):
        return CharacterProperties(sz=round(size * 100), b=False, solidFill="000000",
                                   latin=Font(typeface=EXCEL_FONT))

    def rich(size):
        return RichText(p=[Paragraph(pPr=ParagraphProperties(defRPr=font(size)), endParaRPr=font(size))])

    def extent(artist):
        return artist.get_window_extent(renderer).transformed(inv)

    def place(bb, i, size=True, target=None):
        """A figure-fraction box as a fraction of panel i's chart."""
        return Layout(manualLayout=ManualLayout(
            layoutTarget=target, xMode="edge", yMode="edge", x=bb.x0 * sx, y=(1 - i / n - bb.y1) * n,
            w=bb.width * sx if size else None, h=bb.height * n if size else None))

    def title(text, i=None):
        """Placed where matplotlib has it, or (i None) by Excel."""
        p = Paragraph(pPr=ParagraphProperties(defRPr=font(text.get_fontsize())),
                      r=[RegularTextRun(rPr=font(text.get_fontsize()), t=text.get_text())])
        return Title(tx=Text(rich=RichText(p=[p])), overlay=False,
                     layout=None if i is None else place(extent(text), i, size=False))

    def line(artist):   # a Line2D, or a patch's edge (spine, legend frame)
        c = artist.get_color() if hasattr(artist, "get_color") else artist.get_edgecolor()
        lw = artist.get_linewidth()
        ln = LineProperties(solidFill=colour(c, artist.get_alpha()), w=round(lw * emu_pt))
        if artist.get_linestyle() == "--":   # set to matplotlib's dashes in the XML below
            ln.prstDash = "dash"
        return ln

    def scale(axis, ax, lim, ticks, lo, months=False):
        """Excel axis from matplotlib's ax: range from the last tick <= the data
        to matplotlib's limit; ticks at matplotlib's spacing (months: the step
        whose multiples land on each month's first day)."""
        ticks = np.array([t for t in ticks if lim[0] <= t <= lim[1]])
        if months:
            ticks = ticks[ticks >= ticks[ticks <= lo].max()]
            step, k = np.diff(ticks), np.arange(len(ticks))
            unit = next(u for u in np.arange(step.min(), step.max() + 0.01, 0.01)
                        if (np.floor(ticks[0] + k * u) == ticks).all())
            start = ticks[0]
        else:
            unit = ticks[1] - ticks[0]
            start = ticks[0] + unit * np.floor((lo - ticks[0]) / unit)
            if lim[0] - start > unit / 2:   # too far below matplotlib's limit: half steps
                unit /= 2
                start = ticks[0] + unit * np.floor((lo - ticks[0]) / unit)
        axis.scaling.min, axis.scaling.max, axis.majorUnit = start, lim[1], unit
        axis.delete = False
        axis.majorTickMark = "out"
        axis.txPr = rich(ax.get_xticklabels()[0].get_fontsize())
        axis.spPr = GraphicalProperties(ln=line(ax.spines["bottom"]))
        axis.majorGridlines.spPr = GraphicalProperties(ln=line(ax.xaxis.get_gridlines()[0]))

    with pd.ExcelWriter(path, datetime_format="yyyy-mm-dd") as xl:
        charts = xl.book.create_sheet("charts")
        for i, (ax, s, nights, stays) in enumerate(panels):
            nights.to_excel(xl, sheet_name=str(s), index_label="date")
            stays.to_excel(xl, sheet_name=str(s), startcol=4, index=False)
            ws = xl.sheets[str(s)]

            ch = ScatterChart()
            ch.title = title(ax.title, i)
            ch.y_axis.title = title(ax.yaxis.label)   # Excel's tick labels are wider: let it place this
            ch.layout = place(ax.get_position(), i, target="inner")
            ch.graphical_properties = GraphicalProperties(ln=LineProperties(noFill=True))
            ch.plot_area.graphicalProperties = GraphicalProperties(ln=line(ax.spines["left"]))
            ys = min(nights.min().min(), stays[["recorded_price", "implied_bar"]].min().min())
            xs = min(nights.index.min(), stays["arrival"].min())
            scale(ch.x_axis, ax, np.array(ax.get_xlim()) - offset, ax.get_xticks() - offset,
                  mdates.date2num(xs) - offset, months=True)
            scale(ch.y_axis, ax, ax.get_ylim(), ax.get_yticks(), ys)
            ch.x_axis.number_format = "yyyy-mm"
            if isinstance(ax.yaxis.get_major_formatter(), PercentFormatter):
                ch.y_axis.number_format = '0"%"'

            # Artists in plot order: 2 scatters (stays sheet), then 2 lines (nights sheet)
            sc, ln = ax.collections, ax.lines
            cols = {sc[0]: (5, 6, len(stays)), sc[1]: (5, 7, len(stays)),
                    ln[0]: (1, 3, len(nights)), ln[1]: (1, 2, len(nights))}
            # Excel fills its legend row by row, matplotlib column by column
            for art in (sc[0], ln[0], sc[1], ln[1]):
                x, y, rows = cols[art]
                ser = Series(Reference(ws, min_col=y, min_row=2, max_row=rows + 1),
                             Reference(ws, min_col=x, min_row=2, max_row=rows + 1), title=art.get_label())
                ser.smooth = False
                if art in ln:
                    ser.marker.symbol = "none"
                    ser.graphicalProperties.line = line(art)
                else:
                    # matplotlib's dot: area s (pt²) plus an edge in the same colour
                    size = np.sqrt(art.get_sizes()[0]) + art.get_linewidths()[0]
                    ser.marker.symbol, ser.marker.size = "circle", max(2, round(size))
                    ser.marker.graphicalProperties = GraphicalProperties(
                        solidFill=colour(art.get_facecolor()[0]), ln=LineProperties(noFill=True))
                    ser.graphicalProperties.line.noFill = True
                ch.series.append(ser)

            leg = ax.get_legend()
            frame = leg.get_frame()
            ch.legend.overlay = True
            # Excel needs a little more room than matplotlib for 2 entries per row
            ch.legend.layout = place(extent(leg).expanded(1.25, 1.1), i)
            ch.legend.txPr = rich(leg.get_texts()[0].get_fontsize())
            ch.legend.spPr = GraphicalProperties(solidFill=colour(frame.get_facecolor()), ln=line(frame))

            ch.anchor = AbsoluteAnchor(pos=XDRPoint2D(0, round(i * h / n * emu_in)),
                                       ext=XDRPositiveSize2D(round(cw * emu_in), round(h / n * emu_in)))
            charts.add_chart(ch)

    # openpyxl can't write colour transparency or custom dashes: add them to
    # the chart XML (matplotlib dashes, like Excel's, scale with the line width)
    on, off = plt.rcParams["lines.dashed_pattern"]
    swap = {f'<a:srgbClr val="{c}" />': f'<a:srgbClr val="{c}"><a:alpha val="{round(a * 1e5)}" /></a:srgbClr>'
            for c, a in alphas.items()}
    swap['<a:prstDash val="dash" />'] = f'<a:custDash><a:ds d="{round(on * 1e5)}" sp="{round(off * 1e5)}" /></a:custDash>'
    with zipfile.ZipFile(path) as z:
        files = {f: z.read(f) for f in z.namelist()}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for f, data in files.items():
            if f.startswith("xl/charts/"):
                for old, new in swap.items():
                    data = data.replace(old.encode(), new.encode())
            z.writestr(f, data)


def plot(df, daily, path=PLOT_OUT, pct=False):
    """Prices in EUR, or (pct) in % of the season's reference price p_ref.
    The same plot also goes to an .xlsx of the same name as Excel charts."""
    fig, axes = plt.subplots(len(SEASONS), 1, figsize=(12, 3.8 * len(SEASONS)))
    panels = []
    for ax, s in zip(np.atleast_1d(axes), SEASONS):
        d = daily[daily["season"] == s]
        b = df[(df["season"] == s) & (df["nights"] == 1)]
        ref = p_ref(df, daily, s)
        k = 100 / ref if pct else 1
        rec = recorded_price(df[df["season"] == s], d.index)
        ax.scatter(b["arrival"], b["price"] * k, s=8, color="#bbb", label="Recorded price (1-night stays)")
        ax.scatter(b["arrival"], b["bar"] * k, s=8, color="#2E75B6", alpha=0.6, label="Implied BAR (1-night stays)")
        ax.plot(d.index, rec * k, color="#888", lw=1.2,
                ls="--", label="Old hotel price (recorded, 7-day median)")
        ax.plot(d.index, d["bar"] * k, color="#C0392B", lw=1.4, label="Daily BAR (double)")
        if pct:
            ax.set_title(f"Reconstructed BAR — season {s} (100% = median daily BAR)")
            ax.set_ylabel("% of median daily BAR")
            ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        else:
            ax.set_title(f"Reconstructed BAR — season {s}")
            ax.set_ylabel("EUR / night")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower center", ncol=2)
        panels.append((ax, s, pd.DataFrame({"daily_bar": d["bar"] * k, "old_price_7d": rec * k}),
                       pd.DataFrame({"arrival": b["arrival"], "recorded_price": b["price"] * k,
                                     "implied_bar": b["bar"] * k}).sort_values("arrival")))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    excel_plot(fig, panels, path.replace(".png", ".xlsx"))
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    os.makedirs(BAR_DIR, exist_ok=True)
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
    with open(INPUTS_OUT, "w", encoding="utf-8") as f:
        f.writelines(f"{file_hash(p)}  {p}\n" for p in (XLSX_PATH, RULES_PATH, OCCUPANCY_PATH))
    text = report(df, daily, ladders, rules, occ, hits)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    plot(df, daily)
    plot(df, daily, PLOT_PCT_OUT, pct=True)

    print(text)
    print(f"\n  Saved: {BOOKINGS_OUT}, {DAILY_OUT}, {INPUTS_OUT}, {REPORT_OUT}, {PLOT_OUT}, {PLOT_PCT_OUT} (+ .xlsx charts)")
    print("  Next: python ml_demand_model.py")
