"""Robustness and official baselines for a finished RL run (evaluation only, no
training). Loads the checkpoints saved in results_<season>_<model>_e<eps>_t<steps>/
and evaluates them, with the hotel, the best fixed price, the per-date
schedule and a rooms-left rule, on the TEST seasons:
- in the nominal world, where it must reproduce results.csv and the
  comparison exactly (it evaluates the same agents), with paired significance
  tests against every baseline;
- in perturbed worlds (the other ML model's world, other elasticities, demand
  scaled down or up), every policy deployed unchanged.
The checkpoints must come from the current sim_data.pkl (the elasticity log's
fingerprint). Reads existing files and writes only OUT_DIR."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy import stats

from hotel_season_env import HotelEnv, PRICE_MULTIPLIERS, SIM_DATA_PATH
from rl_agent import (ALGOS, MODELS, MODEL_NAMES, SEASON, ELASTICITY, TIMESTEPS, SEEDS,
                      VAL_SEED0, N_VAL_FIXED, TEST_SEED0, N_TEST, PROTOCOL, ELASTICITY_LOG,
                      run_policy, run_batched, make_envs, hotel_policy, fixed_policy,
                      schedule_policy, greedy, ensemble, summarise, seed_se, find_best_fixed,
                      action_sweep, best_schedule, observed_range, at_grid_edge, steps_tag)


# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR = "robustness_{season}_e{elasticity}_t{steps}"
JOBS    = 6                        # parallel processes (1 = here, same results)

# Worlds the agents are deployed in unchanged: env overrides of the nominal world
# (the agent's own world and forecast). "other" = the other ML model's world
# (its lambda_true and dispersion_k); the agent keeps its own forecast.
WORLDS = {"nominal":     {},
          "other_world": {"demand_model": "other"},
          "eps0.8":      {"elasticity": 0.8},
          "eps1.2":      {"elasticity": 1.2},
          "demand0.85":  {"demand_scale": 0.85},
          "demand1.15":  {"demand_scale": 1.15}}

# Rooms-left rule: one price level per bucket of rooms left tonight, never lower
# with fewer rooms. The buckets of the 23 Oct diagnostic, fixed in advance.
RULE_BUCKETS    = [(0, 2), (3, 4), (5, 7), (8, 10), (11, 13), (14, 99)]
RULE_MAX_PASSES = 10

# Significance: the primary family (mean of seeds vs each baseline, per ML
# model and algorithm) is Holm-corrected; seeds and ensembles are descriptive.
FAMILY_ALPHA = 0.05
BASELINES    = ["hotel", "fixed", "schedule", "rule"]
TERCILES     = ["low", "mid", "high"]


# =============================================================================
# INPUTS: the run's checkpoints, traced to the current world
# =============================================================================

def results_dir(model, args):
    """rl_agent.py's output folder of this run (without --common-world)."""
    return f"results_{args.season}_{model}_e{args.elasticity}_t{steps_tag(args.timesteps)}"


def check_checkpoints(model, args):
    """Fail unless the run's checkpoints come from the current sim_data.pkl.
    The elasticity log's last row of this run (a rerun with the same tag
    overwrites the folder) must carry the current fingerprint and the folder's
    revenues. Returns that row."""
    from ml_demand_model import file_hash         # as append_elasticity_log
    path, folder = ELASTICITY_LOG.format(model=model), results_dir(model, args)
    log = pd.read_csv(path, float_precision="round_trip") if os.path.exists(path) else None
    seeds = " ".join(map(str, args.seeds))
    run = None if log is None else log[
        (log["protocol"] == PROTOCOL) & (log["season"] == args.season)
        & (log["elasticity"] == args.elasticity) & (log["timesteps"] == args.timesteps)
        & (log["seeds"].astype(str) == seeds)]
    if run is None or run.empty:
        raise ValueError(f"{path}: no {PROTOCOL} row for {folder} (seeds {seeds}), so its "
                         "checkpoints can't be traced to a world")
    row, current = run.iloc[-1], file_hash(SIM_DATA_PATH)[:12]
    if row["sim_data"] != current:
        raise ValueError(f"{folder}: its checkpoints were trained in world {row['sim_data']}, "
                         f"but {SIM_DATA_PATH} is {current}. Retrain, or restore that sim_data.pkl")
    table = pd.read_csv(os.path.join(folder, "results.csv"),
                        float_precision="round_trip").set_index("policy")
    n = len(args.seeds)
    pairs = [("hotel_revenue", table["revenue"].iloc[0]),
             ("best_fixed_revenue", table["revenue"].iloc[1])] + [
            (f"{algo.upper()}_mean_revenue", table.loc[f"{name} mean of {n} seeds", "revenue"])
            for algo, (_, _, name) in ALGOS.items()]
    wrong = [col for col, value in pairs if row[col] != value]
    if wrong:
        raise ValueError(f"{path}'s last row of this run ({row['timestamp']}) doesn't match "
                         f"{folder}/results.csv ({', '.join(wrong)})")
    missing = [f"{algo}_seed{s}.zip" for algo in ALGOS for s in args.seeds
               if not os.path.exists(os.path.join(folder, f"{algo}_seed{s}.zip"))]
    if missing:
        raise FileNotFoundError(f"{folder}: missing {', '.join(missing)}")
    return row


def env_kwargs_for(model, world, args):
    """The env of model's pipeline in world: rl_agent's env_kwargs plus the overrides."""
    kw = {**dict(season=args.season, demand_model=model, forecast_model=model,
                 elasticity=args.elasticity), **WORLDS[world]}
    if kw["demand_model"] == "other":
        kw["demand_model"] = next(m for m in MODELS if m != model)
    return kw


def with_ensemble(seeds):
    return len(seeds) >= 3 and len(seeds) % 2 == 1      # as rl_agent.run_pipeline


# =============================================================================
# ROOMS-LEFT RULE AND POLICY MAP
# =============================================================================

def rooms_left(obs, capacity):
    """Rooms left tonight, read from the observation (committed rooms tonight,
    scaled to [-1, 1]); exact, since committed rooms are whole numbers."""
    committed = (obs[:, 1].astype(np.float64) + 1) * capacity / 2
    if np.abs(committed - np.rint(committed)).max() > 1e-3:
        raise ValueError("The observation's committed rooms are not whole numbers")
    return capacity - np.rint(committed).astype(int)


def rule_predict(levels, capacity):
    """The rooms-left rule as a batched policy for run_batched: the price level
    of the bucket of tonight's rooms left. It reads only obs[1], which the
    agents see too."""
    levels, tops = np.asarray(levels), [hi for _, hi in RULE_BUCKETS]
    return lambda obs: levels[np.searchsorted(tops, rooms_left(obs, capacity))]


def choose_rule(env_kwargs, start):
    """The rooms-left rule chosen on the validation seasons, where
    find_best_fixed chooses the best fixed price: coordinate ascent over
    monotone levels, starting from the best fixed level (start) in every
    bucket, so on validation it can only beat best fixed. Buckets are tried
    from the scarcest; a level moves only on a strict improvement (ties keep
    it). Returns (levels, its validation run, passes, converged)."""
    envs = make_envs(env_kwargs, N_VAL_FIXED)
    capacity, top, runs = envs[0].capacity, len(PRICE_MULTIPLIERS) - 1, {}

    def score(levels):
        if levels not in runs:
            runs[levels] = run_batched(envs, rule_predict(levels, capacity), VAL_SEED0)[0]
        return runs[levels]["reward"].mean()

    levels = (start,) * len(RULE_BUCKETS)
    for n_pass in range(1, RULE_MAX_PASSES + 1):
        moved = False
        for b in range(len(levels)):
            hi = levels[b - 1] if b > 0 else top               # monotone: within the neighbours
            lo = levels[b + 1] if b < len(levels) - 1 else 0
            choice = levels
            for a in range(lo, hi + 1):
                cand = levels[:b] + (a,) + levels[b + 1:]
                if score(cand) > score(choice):
                    choice = cand
            moved |= choice != levels
            levels = choice
        if not moved:
            return levels, runs[levels], n_pass, True
    return levels, runs[levels], RULE_MAX_PASSES, False


def forecast_terciles(forecast):
    """Tercile of each date's forecast over the season (0 low, 1 mid, 2 high)."""
    return pd.qcut(pd.Series(forecast).rank(method="first"), 3, labels=False).to_numpy()


def recording(predict, counts, terciles, capacity):
    """predict, also counting each day's (rooms left tonight, forecast tercile,
    action) in counts. run_batched calls it once per day, so the counts come
    from the rollout that is scored."""
    day = [0]

    def wrapped(obs):
        actions = np.asarray(predict(obs))
        np.add.at(counts, (rooms_left(obs, capacity), terciles[day[0]], actions), 1)
        day[0] += 1
        return actions
    return wrapped


# =============================================================================
# EVALUATION TASKS (top level: worker processes import them)
# =============================================================================

def sweep_task(model, args):
    """Benchmarks chosen in model's nominal world, as rl_agent.run_pipeline
    chooses them: best fixed and per-date schedules, also within the hotel's range."""
    torch.set_num_threads(1)
    env = HotelEnv(**env_kwargs_for(model, "nominal", args))
    best, fixed_val = find_best_fixed(env)
    hotel_val, _ = run_policy(env, hotel_policy, n=N_VAL_FIXED, seed0=VAL_SEED0)
    sweep, in_range = action_sweep(env), observed_range(env)
    return dict(best=best, best_in=find_best_fixed(env, actions=in_range)[0],
                schedule=best_schedule(sweep, range(len(PRICE_MULTIPLIERS))),
                schedule_in=best_schedule(sweep, in_range),
                fixed_val=fixed_val["reward"].mean(), hotel_val=hotel_val["reward"].mean())


def rule_task(model, args):
    """The rooms-left rule chosen in model's nominal world (on validation)."""
    torch.set_num_threads(1)
    kw = env_kwargs_for(model, "nominal", args)
    best, _ = find_best_fixed(HotelEnv(**kw))
    levels, val, passes, converged = choose_rule(kw, best)
    return dict(rule=levels, rule_val=val["reward"].mean(), rule_passes=passes,
                rule_converged=converged, rule_start=best)


def evaluate_world(model, world, bench, args):
    """Every policy of model's pipeline on the test seasons of world, deployed
    unchanged (bench: chosen in the nominal world). Returns the per-season
    runs, the policy-map counts and the best fixed level re-chosen in world."""
    torch.set_num_threads(1)
    kw = env_kwargs_for(model, world, args)
    env, envs = HotelEnv(**kw), make_envs(kw, N_TEST)
    terciles, cap, n_act = forecast_terciles(env.forecast), env.capacity, len(PRICE_MULTIPLIERS)
    test = dict(n=N_TEST, seed0=TEST_SEED0)
    runs, maps = {}, {}

    runs["hotel"] = run_policy(env, hotel_policy, **test)[0]
    runs["fixed"] = run_policy(env, fixed_policy(bench["best"]), **test)[0]
    rechosen = bench["best"] if world == "nominal" else find_best_fixed(env)[0]
    runs["rechosen"] = runs["fixed"] if rechosen == bench["best"] else \
        run_policy(env, fixed_policy(rechosen), **test)[0]
    runs["schedule"] = run_policy(env, schedule_policy(bench["schedule"]), **test)[0]
    if world == "nominal":
        runs["fixed_in"] = run_policy(env, fixed_policy(bench["best_in"]), **test)[0]
        runs["schedule_in"] = run_policy(env, schedule_policy(bench["schedule_in"]), **test)[0]

    def batched(key, predict):
        maps[key] = np.zeros((cap + 1, len(TERCILES), n_act), dtype=int)
        runs[key] = run_batched(envs, recording(predict, maps[key], terciles, cap), TEST_SEED0)[0]

    batched("rule", rule_predict(bench["rule"], cap))
    folder = results_dir(model, args)
    for algo, (cls, _, _) in ALGOS.items():
        agents = [cls.load(os.path.join(folder, f"{algo}_seed{s}"), device="cpu") for s in args.seeds]
        for s, agent in zip(args.seeds, agents):
            batched((algo, s), greedy(agent))
        if with_ensemble(args.seeds):
            batched((algo, "ensemble"), ensemble(agents))
    return dict(runs=runs, maps=maps, rechosen=rechosen, kwargs=kw)


def run_tasks(jobs, tasks):
    """[(function, args)] -> their results, in parallel processes or here."""
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
            futures = [pool.submit(fn, *a) for fn, a in tasks]
            return [f.result() for f in futures]
    return [fn(*a) for fn, a in tasks]


# =============================================================================
# STATISTICS
# =============================================================================

def paired(df, base):
    """Gain over base (% of base's mean revenue), its paired SE over the
    seasons and the normal 95% CI half-width, as summarise computes them."""
    diff = df["reward"].values - base["reward"].values
    ref = base["reward"].mean()
    se = 100 * diff.std(ddof=1) / np.sqrt(len(diff)) / ref
    return 100 * diff.mean() / ref, se, stats.norm.ppf(0.975) * se


def compare(runs, base):
    """paired for one run; for a list of seed runs, the gain of their mean with
    seed_se (the variance between seeds plus the season noise, t(n-1) CI)."""
    if isinstance(runs, list):
        return (paired(pd.concat(runs).groupby(level=0).mean(), base)[0], *seed_se(runs, base))
    return paired(runs, base)


def holm(p):
    """Holm's step-down adjusted p-values: family-wise error rate under any dependence."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty(len(p))
    out[order] = np.minimum(adj, 1.0)
    return out


def test_row(gain, se, ci, df):
    """t (or z, df = inf) and its two-sided p, and the CI ends."""
    t = gain / se if se > 0 else np.nan
    p = 2 * (stats.norm.sf(abs(t)) if np.isinf(df) else stats.t.sf(abs(t), df))
    return {"diff_%": gain, "se_%": se, "df": df, "t": t, "p": p,
            "ci95_lo_%": gain - ci, "ci95_hi_%": gain + ci}


# =============================================================================
# TABLES
# =============================================================================

def names_of(bench, n):
    """Display names of the runs' keys, as in results.csv. The re-chosen best
    fixed level differs between worlds: robustness.csv has it in a column."""
    mult = lambda a: f"{PRICE_MULTIPLIERS[a]:.2f}x"
    names = {"hotel": "Hotel (actual prices)",
             "fixed": f"Best fixed ({mult(bench['best'])})",
             "rechosen": "Best fixed re-chosen",
             "schedule": "Per-date schedule (benchmark)",
             "fixed_in": f"Best fixed in hotel's range ({mult(bench['best_in'])})",
             "schedule_in": "Per-date schedule in hotel's range",
             "rule": "Rooms-left rule"}
    for algo, (_, _, name) in ALGOS.items():
        names[algo, "mean"] = f"{name} mean of {n} seeds"
        names[algo, "ensemble"] = f"{name} ensemble (median of {n} seeds)"
    return lambda key: names.get(key, f"{ALGOS[key[0]][2]} seed {key[1]}" if isinstance(key, tuple) else key)


def seed_runs(runs, algo, seeds):
    return [runs[algo, s] for s in seeds]


def rebuild_results(model, res, bench, env, args):
    """results.csv's rows and the comparison's columns, rebuilt as
    rl_agent.run_pipeline builds them, from the nominal re-evaluation."""
    runs, n, name = res["runs"], len(args.seeds), names_of(bench, len(args.seeds))
    hotel, fixed = runs["hotel"], runs["fixed"]
    rows = [summarise(name("hotel"), hotel, hotel, env),
            {**summarise(name("fixed"), fixed, hotel, env),
             "val_vs_hotel_%": 100 * (bench["fixed_val"] / bench["hotel_val"] - 1)},
            summarise(name("schedule"), runs["schedule"], hotel, env),
            summarise(name("fixed_in"), runs["fixed_in"], hotel, env),
            summarise(name("schedule_in"), runs["schedule_in"], hotel, env)]
    a, comp = env.actual, []
    for algo, (_, _, algo_name) in ALGOS.items():
        dfs = seed_runs(runs, algo, args.seeds)
        rows += [summarise(name((algo, s)), runs[algo, s], hotel, env) for s in args.seeds]
        mean_df = pd.concat(dfs).groupby(level=0).mean()
        mean_row = summarise(name((algo, "mean")), mean_df, hotel, env)
        mean_row["vs_hotel_se_%"], mean_row["vs_hotel_ci95_%"] = seed_se(dfs, hotel)
        rows.append(mean_row)
        ens_gain = np.nan
        if with_ensemble(args.seeds):
            ens = runs[algo, "ensemble"]
            rows.append(summarise(name((algo, "ensemble")), ens, hotel, env))
            ens_gain = 100 * (ens["reward"].mean() / fixed["reward"].mean() - 1)
        dyn_se, dyn_ci = seed_se(dfs, fixed)
        comp.append({
            "ml_model": MODEL_NAMES[model], "algo": algo_name,
            "sim_vs_real_revenue_%": 100 * (hotel["revenue"].mean() / a["revenue"] - 1),
            "sim_vs_real_bookings_%": 100 * (hotel["accepted"].mean() / a["confirmed"] - 1),
            "hotel_revenue": hotel["revenue"].mean(),
            "best_fixed_vs_hotel_%": 100 * (fixed["reward"].mean() / hotel["reward"].mean() - 1),
            "rl_revenue": mean_df["revenue"].mean(),
            "rl_vs_hotel_%": mean_row["vs_hotel_%"],
            "rl_vs_hotel_se_%": mean_row["vs_hotel_se_%"],
            "rl_vs_hotel_ci95_%": mean_row["vs_hotel_ci95_%"],
            "rl_vs_best_fixed_%": 100 * (mean_df["reward"].mean() / fixed["reward"].mean() - 1),
            "rl_vs_best_fixed_se_%": dyn_se, "rl_vs_best_fixed_ci95_%": dyn_ci,
            "rl_seed_spread_%": (100 * np.std([d["reward"].mean() for d in dfs], ddof=1)
                                 / hotel["reward"].mean() if n > 1 else np.nan),
            "schedule_vs_best_fixed_%": 100 * (runs["schedule"]["reward"].mean()
                                               / fixed["reward"].mean() - 1),
            "best_fixed_in_range_vs_hotel_%": 100 * (runs["fixed_in"]["reward"].mean()
                                                     / hotel["reward"].mean() - 1),
            "ensemble_vs_best_fixed_%": ens_gain})
    return rows, comp


def check_reproduction(model, rows, comp, args):
    """Every rebuilt value against results.csv and the comparison CSV: they
    must be equal (exactly, as parsed back from the CSVs). Returns the table."""
    same = lambda x, y: (pd.isna(x) and pd.isna(y)) or x == y
    out = []
    table = pd.read_csv(os.path.join(results_dir(model, args), "results.csv"),
                        float_precision="round_trip").set_index("policy")
    for r in rows:
        for col, v in r.items():
            if col != "policy":
                repo = table.loc[r["policy"], col]
                out.append(("results", r["policy"], col, repo, v))
    path = (f"comparison_{args.season}_e{args.elasticity}_t{steps_tag(args.timesteps)}.csv")
    cmp_ = pd.read_csv(path, float_precision="round_trip").set_index(["ml_model", "algo"])
    for r in comp:
        for col, v in r.items():
            if col not in ("ml_model", "algo"):
                out.append(("comparison", f"{r['ml_model']} + {r['algo']}", col,
                            cmp_.loc[(r["ml_model"], r["algo"]), col], v))
    return pd.DataFrame([{"forecaster": model, "source": s, "policy": p, "column": c,
                          "repo_value": x, "rebuilt_value": y,
                          "abs_diff": abs(x - y) if not pd.isna(x) else np.nan, "match": same(x, y)}
                         for s, p, c, x, y in out])


def robustness_rows(model, world, res, bench, args):
    """One row per policy in world: its revenue and occupancy, and its gain
    over the hotel, best fixed (nominal level and re-chosen) and the rule."""
    runs, n = res["runs"], len(args.seeds)
    name, env, kw = names_of(bench, n), HotelEnv(**res["kwargs"]), res["kwargs"]
    bases = {"hotel": runs["hotel"], "best_fixed": runs["fixed"],
             "best_fixed_rechosen": runs["rechosen"], "rule": runs["rule"]}
    keys = ["hotel", "fixed", "rechosen", "schedule", "rule"]
    for algo in ALGOS:
        keys += [(algo, s) for s in args.seeds] + [(algo, "mean")] + \
                ([(algo, "ensemble")] if with_ensemble(args.seeds) else [])
    rows = []
    for key in keys:
        algo, seed = (key if isinstance(key, tuple) else ("", ""))
        if seed == "mean":
            run = seed_runs(runs, algo, args.seeds)
            df = pd.concat(run).groupby(level=0).mean()
        else:
            run = df = runs[key]
        s = summarise(name(key), df, runs["hotel"], env)
        row = {"forecaster": model, "world": world, "demand_model": kw["demand_model"],
               "elasticity": kw["elasticity"], "demand_scale": kw.get("demand_scale", 1.0),
               "policy": s["policy"], "algo": ALGOS[algo][2] if algo else "",
               "seed": seed if isinstance(seed, int) else "",
               **{c: s[c] for c in ("revenue", "room_nights", "occupancy_%", "ADR",
                                    "lost_requests", "above_range_%")}}
        for b, base in bases.items():
            row[f"vs_{b}_%"], row[f"vs_{b}_se_%"], row[f"vs_{b}_ci95_%"] = compare(run, base)
        row["best_fixed_multiplier"] = PRICE_MULTIPLIERS[bench["best"]]
        row["best_fixed_rechosen_multiplier"] = PRICE_MULTIPLIERS[res["rechosen"]]
        rows.append(row)
    return rows


def significance_rows(model, res, bench, args):
    """Paired tests on the test seasons of each algorithm (mean of seeds:
    primary; each seed and the ensemble: descriptive) against each baseline."""
    runs, n = res["runs"], len(args.seeds)
    name = names_of(bench, n)
    rows = []
    for algo, (_, _, algo_name) in ALGOS.items():
        cands = [((algo, "mean"), "mean_of_seeds", seed_runs(runs, algo, args.seeds), n - 1)]
        cands += [((algo, s), "seed", runs[algo, s], np.inf) for s in args.seeds]
        if with_ensemble(args.seeds):
            cands.append(((algo, "ensemble"), "ensemble", runs[algo, "ensemble"], np.inf))
        for key, kind, run, df in cands:
            for b in BASELINES:
                rows.append({"forecaster": model, "algo": algo_name, "policy": name(key),
                             "kind": kind, "baseline": name(b),
                             **test_row(*compare(run, runs[b]), df),
                             "family": "primary" if kind == "mean_of_seeds" else "secondary"})
    return rows


def policy_map_rows(model, world, res, bench, args):
    """Tidy policy map: per policy, rooms left tonight and forecast tercile,
    the days at each multiplier (every multiplier of each visited cell)."""
    name = names_of(bench, len(args.seeds))
    rows = []
    for key, counts in res["maps"].items():
        algo, seed = key if isinstance(key, tuple) else ("", "")
        cells = counts.sum(axis=2)
        for r, t in zip(*np.nonzero(cells)):
            for a, mult in enumerate(PRICE_MULTIPLIERS):
                rows.append({"forecaster": model, "world": world, "policy": name(key),
                             "algo": ALGOS[algo][2] if algo else "", "seed": seed,
                             "rooms_left": r, "forecast_tercile": TERCILES[t], "multiplier": mult,
                             "days": counts[r, t, a], "cell_days": cells[r, t],
                             "share": counts[r, t, a] / cells[r, t]})
    return rows


def per_season_rows(model, world, res, bench, args):
    name = names_of(bench, len(args.seeds))
    frames = []
    for key, df in res["runs"].items():
        algo, seed = key if isinstance(key, tuple) else ("", "")
        frames.append(df.assign(forecaster=model, world=world, policy=name(key),
                                algo=ALGOS[algo][2] if algo else "", seed=seed,
                                season_seed=TEST_SEED0 + df.index))
    cols = ["forecaster", "world", "policy", "algo", "seed", "season_seed", "reward", "revenue",
            "room_nights", "accepted", "cancelled", "lost", "above_range"]
    return pd.concat(frames)[cols]


# =============================================================================
# MAIN
# =============================================================================

def evaluate(args):
    """1. The benchmarks and the rule, chosen in each nominal world (validation);
    2. every policy in every world (test). Returns (bench, res)."""
    tasks = [(sweep_task, (m, args)) for m in args.models] + \
            [(rule_task, (m, args)) for m in args.models]
    done = run_tasks(args.jobs, tasks)
    k = len(args.models)
    bench = {m: {**done[i], **done[k + i]} for i, m in enumerate(args.models)}
    for m, b in bench.items():
        if b["rule_start"] != b["best"]:
            raise RuntimeError(f"{m}: the rule search and the sweep chose different best fixed levels")

    tasks = [(evaluate_world, (m, w, bench[m], args)) for m in args.models for w in WORLDS]
    res = dict(zip([(m, w) for m in args.models for w in WORLDS], run_tasks(args.jobs, tasks)))
    return bench, res


def report(bench, res, log_rows, args):
    """3. Check the nominal world against the run; 4. write the tables."""
    out_dir = OUT_DIR.format(season=args.season, elasticity=args.elasticity,
                             steps=steps_tag(args.timesteps))
    log = []

    def out(*lines):
        for line in lines:
            print(line); log.append(line)
        print(); log.append("")

    fmt = lambda v: f"{v:,.2f}"
    out("=" * 72,
        f" Robustness and baselines: season {args.season}, elasticity {args.elasticity}, "
        f"{args.timesteps:,} steps, seeds {' '.join(map(str, args.seeds))}",
        *[f" {results_dir(m, args)}: checkpoints from world {r['sim_data']} "
          f"(the current {SIM_DATA_PATH}; elasticity log row {r['timestamp']})"
          for m, r in log_rows.items()],
        f" Test seasons {TEST_SEED0}-{TEST_SEED0 + N_TEST - 1}; benchmarks and the rule "
        f"chosen on validation seasons {VAL_SEED0}-{VAL_SEED0 + N_VAL_FIXED - 1} "
        "of the nominal world",
        "=" * 72)

    # 3. The nominal world must reproduce results.csv and the comparison exactly
    os.makedirs(out_dir, exist_ok=True)
    repro, base_rows, rule_rows = [], [], []
    for m in args.models:
        env = HotelEnv(**env_kwargs_for(m, "nominal", args))
        rebuilt, comp = rebuild_results(m, res[m, "nominal"], bench[m], env, args)
        repro.append(check_reproduction(m, rebuilt, comp, args))

        # Thesis table: the nominal benchmarks, the rule and each algorithm
        runs, b = res[m, "nominal"]["runs"], bench[m]
        name = names_of(b, len(args.seeds))
        hotel, fixed = runs["hotel"], runs["fixed"]
        by_policy = {r["policy"]: r for r in rebuilt}
        results = pd.read_csv(os.path.join(results_dir(m, args), "results.csv"),
                              float_precision="round_trip").set_index("policy")
        rule_row = {**summarise(name("rule"), runs["rule"], hotel, env),
                    "val_vs_hotel_%": 100 * (b["rule_val"] / b["hotel_val"] - 1)}
        keys = ["hotel", "fixed", "fixed_in", "schedule", "schedule_in", "rule"]
        for algo in ALGOS:
            keys += [(algo, "mean")] + ([(algo, "ensemble")] if with_ensemble(args.seeds) else [])
        for key in keys:
            r = rule_row if key == "rule" else dict(by_policy[name(key)])
            if isinstance(key, tuple) and key[1] == "ensemble":   # chosen by rl_agent on validation
                r["val_vs_hotel_%"] = results.loc[r["policy"], "val_vs_hotel_%"]
            run = seed_runs(runs, key[0], args.seeds) if isinstance(key, tuple) and key[1] == "mean" \
                else runs[key]
            gain, se, ci = compare(run, fixed)
            base_rows.append({"forecaster": m, **{c: v for c, v in r.items() if c != "val_vs_hotel_%"},
                              "vs_best_fixed_%": gain, "vs_best_fixed_se_%": se,
                              "vs_best_fixed_ci95_%": ci, "val_vs_hotel_%": r.get("val_vs_hotel_%")})

        val_fixed = 100 * (b["rule_val"] / b["fixed_val"] - 1)
        for (lo, hi), a in zip(RULE_BUCKETS, b["rule"]):
            rule_rows.append({"world_model": m, "rooms_left_lo": lo, "rooms_left_hi": min(hi, env.capacity),
                              "multiplier": PRICE_MULTIPLIERS[a], "price_eur": env.prices[a],
                              "val_vs_hotel_%": rule_row["val_vs_hotel_%"],
                              "val_vs_best_fixed_%": val_fixed, "passes": b["rule_passes"],
                              "converged": b["rule_converged"]})

    repro = pd.concat(repro, ignore_index=True)
    repro.to_csv(os.path.join(out_dir, "reproduction.csv"), index=False)
    bad = repro[~repro["match"]]
    if len(bad):
        raise RuntimeError(f"The re-evaluation does not reproduce the run ({len(bad)} values differ, "
                           f"largest {bad['abs_diff'].max():.3g}): see {out_dir}/reproduction.csv")
    out(f"Reproduction: all {len(repro)} values of results.csv and the comparison rebuilt exactly "
        "from the checkpoints (same agents, same seasons)")

    # 4. Tables
    rules = pd.DataFrame(rule_rows)
    rules.to_csv(os.path.join(out_dir, "rooms_left_rule.csv"), index=False)
    for m in args.models:
        r = rules[rules["world_model"] == m]
        out(f"Rooms-left rule, {MODEL_NAMES[m]} world (chosen on {N_VAL_FIXED} validation seasons, "
            f"{r['passes'].iloc[0]} passes): " + ", ".join(
                f"{lo}-{hi} rooms {mult:.2f}x" for lo, hi, mult in
                zip(r["rooms_left_lo"], r["rooms_left_hi"], r["multiplier"]))
            + f"; validation {r['val_vs_hotel_%'].iloc[0]:+.2f}% vs hotel, "
              f"{r['val_vs_best_fixed_%'].iloc[0]:+.2f}% vs best fixed",
            *([f"  !!! The rule search did not converge in {RULE_MAX_PASSES} passes"]
              if not r["converged"].iloc[0] else []))

    base = pd.DataFrame(base_rows)
    base.to_csv(os.path.join(out_dir, "baselines.csv"), index=False)
    out("Baselines and agents on the TEST seasons (nominal world; mean-of-seeds SEs include "
        "the variance between seeds):",
        base[["forecaster", "policy", "revenue", "vs_hotel_%", "vs_hotel_ci95_%", "vs_best_fixed_%",
              "vs_best_fixed_ci95_%", "occupancy_%", "ADR", "lost_requests", "above_range_%"]]
        .to_string(index=False, float_format=fmt))

    sig = pd.DataFrame([r for m in args.models for r in significance_rows(m, res[m, "nominal"], bench[m], args)])
    primary = sig["family"] == "primary"
    sig["p_holm"] = np.nan
    sig.loc[primary, "p_holm"] = holm(sig.loc[primary, "p"])
    sig["significant"] = np.where(primary, sig["p_holm"] < FAMILY_ALPHA, sig["p"] < FAMILY_ALPHA)
    sig.to_csv(os.path.join(out_dir, "significance.csv"), index=False)
    out(f"Significance on the {N_TEST} paired test seasons. Primary family ({primary.sum()} tests, "
        f"Holm, FWER {FAMILY_ALPHA:.0%}): mean of seeds, SE with the variance between seeds, "
        f"t({len(args.seeds) - 1}); seeds and ensembles: paired z, unadjusted (descriptive).",
        sig[primary][["forecaster", "policy", "baseline", "diff_%", "ci95_lo_%", "ci95_hi_%", "p",
                      "p_holm", "significant"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    rob = pd.DataFrame([r for (m, w), x in res.items() for r in robustness_rows(m, w, x, bench[m], args)])
    rob.to_csv(os.path.join(out_dir, "robustness.csv"), index=False)
    view = rob[rob["seed"].eq("")]                    # benchmarks, means and ensembles
    for col, label in [("vs_hotel_%", "vs hotel"), ("vs_best_fixed_%", "vs best fixed (nominal level)"),
                       ("vs_best_fixed_rechosen_%", "vs best fixed re-chosen in each world")]:
        for m in args.models:
            v = view[view["forecaster"] == m]
            piv = (v.pivot(index="policy", columns="world", values=col)
                    .reindex(index=v["policy"].unique(), columns=list(WORLDS)))
            out(f"Robustness, {MODEL_NAMES[m]} pipeline, % {label} (test seasons, every policy "
                "deployed unchanged):", piv.to_string(float_format=fmt))
    for (m, w), x in res.items():
        if w != "nominal" and at_grid_edge(x["rechosen"]):
            out(f"  !!! {MODEL_NAMES[m]}, {w}: the re-chosen best fixed price is at the edge of "
                f"the price grid ({PRICE_MULTIPLIERS[x['rechosen']]:.2f}x)")

    pd.DataFrame([r for (m, w), x in res.items() for r in policy_map_rows(m, w, x, bench[m], args)]) \
        .to_csv(os.path.join(out_dir, "policy_map.csv"), index=False)
    pd.concat([per_season_rows(m, w, x, bench[m], args) for (m, w), x in res.items()]) \
        .to_csv(os.path.join(out_dir, "per_season.csv"), index=False)
    out(f"Saved {out_dir}/: baselines.csv, significance.csv, robustness.csv, policy_map.csv, "
        "rooms_left_rule.csv, per_season.csv, reproduction.csv, summary.txt")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--elasticity", type=float, default=ELASTICITY)
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--jobs", type=int, default=JOBS, help="parallel processes (1 = sequential)")
    args = ap.parse_args()
    log_rows = {m: check_checkpoints(m, args) for m in args.models}   # fail before any work
    print(f"Evaluating {', '.join(results_dir(m, args) for m in args.models)} in "
          f"{len(WORLDS)} worlds ({args.jobs} processes)...")
    report(*evaluate(args), log_rows, args)


if __name__ == "__main__":
    main()
