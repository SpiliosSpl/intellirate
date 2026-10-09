import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy

from hotel_season_env import HotelEnv, PRICE_MULTIPLIERS


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON      = 2026
MODELS      = ["rf", "xgb"]
MODEL_NAMES = {"rf": "ML1 RF", "xgb": "ML2 XGB"}
ELASTICITY  = 1.0
CAPACITY    = None                 # None -> the season's STD rooms (2025: 18, 2026: 17)
TIMESTEPS   = 300_000              # per algorithm and seed
SEEDS       = [42, 7, 123]
JOBS        = 3                    # parallel processes: the seeds of one ML x RL train together
N_EVAL      = 200                  # simulated seasons per policy
EVAL_SEED0  = 100_000              # evaluation seasons: never seen in training

PPO_PARAMS = dict(learning_rate=3e-4, n_steps=2048, batch_size=256,
                  n_epochs=10, gamma=0.99, ent_coef=0.01)

# DQN: every setting is written out (not left to SB3 defaults) so that the
# Double DQN added later can reuse exactly this dict. The only difference
# between the two is then the TD target (online net selects, target net
# evaluates), a subclass of DQN overriding train(). Same network size as PPO.
DQN_PARAMS = dict(learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000,
                  batch_size=128, gamma=0.99, train_freq=4, gradient_steps=1,
                  target_update_interval=2_000, tau=1.0,
                  exploration_fraction=0.3, exploration_initial_eps=1.0,
                  exploration_final_eps=0.02, max_grad_norm=10,
                  policy_kwargs=dict(net_arch=[64, 64]))

# RL algorithms: key -> (SB3 class, hyperparameters, display name).
# Double DQN later: "ddqn": (DoubleDQN, DQN_PARAMS, "Double DQN")
ALGOS = {
    "ppo": (PPO, PPO_PARAMS, "PPO"),
    "dqn": (DQN, DQN_PARAMS, "DQN"),
}
RL_COLOURS = ["#2E75B6", "#8E44AD", "#16A085", "#7F8C8D"]   # by position in --algos

SENSITIVITY_ELASTICITIES = [0.6, 0.8, 1.0, 1.2, 1.5]
SENSITIVITY_N_EVAL       = 100     # seasons per policy (same for hotel and fixed)


# =============================================================================
# POLICY EVALUATION
# =============================================================================

def run_policy(env, act, n=N_EVAL, seed0=EVAL_SEED0, record_path=False):
    """act(env, obs) -> result of env.step / env.step_price."""
    rows, path = [], None
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        tot = dict(revenue=0.0, reward=0.0, room_nights=0, accepted=0,
                   cancelled=0, lost=0)
        trace, done = [], False
        while not done:
            obs, reward, done, _, info = act(env, obs)
            tot["reward"] += reward
            for key in ("revenue", "room_nights", "accepted", "cancelled", "lost"):
                tot[key] += info[key]
            if record_path and i == 0:
                trace.append((info["date"], info["price"], info["occupied"]))
        rows.append(tot)
        if record_path and i == 0:
            path = pd.DataFrame(trace, columns=["date", "price", "occupied"])
    return pd.DataFrame(rows), path


def hotel_policy(env, obs):
    return env.step_price(env.hotel_price[env.t])


def fixed_policy(action):
    return lambda env, obs: env.step(action)


def agent_policy(model):
    """Greedy policy of a trained SB3 agent (PPO: mode, DQN: argmax Q)."""
    return lambda env, obs: env.step(int(model.predict(obs, deterministic=True)[0]))


def summarise(name, df, base, env):
    diff = df["reward"].values - base["reward"].values          # paired
    days = env.n_days
    return {
        "policy": name,
        "revenue": df["revenue"].mean(),
        "vs_hotel_%": 100 * diff.mean() / base["reward"].mean(),
        "vs_hotel_se_%": 100 * diff.std(ddof=1) / np.sqrt(len(diff)) / base["reward"].mean(),
        "room_nights": df["room_nights"].mean(),
        "occupancy_%": 100 * df["room_nights"].mean() / (env.capacity * days),
        "ADR": df["revenue"].mean() / df["room_nights"].mean(),
        "RevPAR": df["revenue"].mean() / (env.capacity * days),
        "lost_requests": df["lost"].mean(),
    }


def find_best_fixed(env, n=N_EVAL):
    """Best constant price level: (its action, mean reward of every level,
    its evaluation seasons)."""
    runs = [run_policy(env, fixed_policy(a), n=n)[0] for a in range(len(PRICE_MULTIPLIERS))]
    means = [df["reward"].mean() for df in runs]
    best = int(np.argmax(means))
    return best, means, runs[best]


def train_agent(algo, seed, log_dir, env_kwargs, timesteps):
    cls, params, _ = ALGOS[algo]
    os.makedirs(log_dir, exist_ok=True)
    # Reward normalisation: off-policy agents store raw rewards in the replay
    # buffer and normalise them when sampled, so this is valid for DQN too.
    venv = VecNormalize(DummyVecEnv([lambda: Monitor(HotelEnv(**env_kwargs), log_dir)]),
                        norm_obs=False, norm_reward=True, gamma=params["gamma"])
    model = cls("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **params)
    model.learn(total_timesteps=timesteps)
    return model


def train_and_evaluate(algo, seed, log_dir, save_path, env_kwargs, timesteps):
    """One seed: train, save, evaluate. Top-level so that worker processes
    (spawned on Windows) can import it. One torch thread per process: the
    network is too small to gain from more, and it measured faster."""
    torch.set_num_threads(1)
    model = train_agent(algo, seed, log_dir, env_kwargs, timesteps)
    model.save(save_path)
    return run_policy(HotelEnv(**env_kwargs), agent_policy(model), record_path=True)


# =============================================================================
# PLOTS
# =============================================================================

def plot_learning(log_dirs, seeds, algo_name, hotel_mean, fixed_mean, path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for d, sd in zip(log_dirs, seeds):
        x, y = ts2xy(load_results(d), "timesteps")
        w = max(1, len(y) // 30)
        ax.plot(x[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid"), lw=1.8,
                label=f"{algo_name} seed {sd}")
    ax.axhline(hotel_mean, color="#C0392B", ls="--", lw=1.5, label="Hotel (actual prices)")
    ax.axhline(fixed_mean, color="#E67E22", ls=":", lw=1.8, label="Best fixed price")
    ax.set_xlabel("Training timesteps (days simulated)")
    ax.set_ylabel("Season revenue (EUR)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_price_path(env, hotel_path, rl_paths, path, title):
    """rl_paths: {algorithm display name: price path of one simulated season}."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.step(hotel_path["date"], hotel_path["price"], where="mid", color="#C0392B",
             lw=1.4, label="Hotel BAR (reconstructed)")
    for colour, (name, p) in zip(RL_COLOURS, rl_paths.items()):
        ax1.step(p["date"], p["price"], where="mid", color=colour,
                 lw=1.4, label=f"{name} price")
    ax1.set_ylabel("BAR, double occupancy (EUR / night)")
    ax2 = ax1.twinx()
    ax2.fill_between(env.dates, env.forecast, color="grey", alpha=0.2,
                     label="Demand forecast (requests/day)")
    ax2.set_ylabel("Forecast requests / day")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax1.set_title(title)
    ax1.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_comparison(comp, path, title):
    """Grouped bars: one group per ML model; best fixed price + one bar per RL algorithm."""
    groups = list(dict.fromkeys(comp["ml_model"]))
    algos = list(dict.fromkeys(comp["algo"]))
    n_bars = 1 + len(algos)
    w = 0.8 / n_bars
    x = np.arange(len(groups))
    offset = lambda j: (j - (n_bars - 1) / 2) * w

    fig, ax = plt.subplots(figsize=(4 + 2.5 * len(groups), 5))
    fixed = comp.groupby("ml_model", sort=False)["best_fixed_vs_hotel_%"].first().reindex(groups)
    ax.bar(x + offset(0), fixed, w, color="#E67E22", label="Best fixed price")
    for j, (algo, colour) in enumerate(zip(algos, RL_COLOURS), start=1):
        sub = comp[comp["algo"] == algo].set_index("ml_model").reindex(groups)
        ax.bar(x + offset(j), sub["rl_vs_hotel_%"], w, yerr=sub["rl_vs_hotel_se_%"],
               color=colour, capsize=4, label=f"{algo} (mean of seeds)")
    ax.axhline(0, color="#C0392B", lw=1.2, ls="--", label="Hotel (actual prices)")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Revenue vs hotel (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# =============================================================================
# ONE PIPELINE:  ML model + every RL algorithm
# =============================================================================

def run_pipeline(model_name, args):
    world = args.common_world or model_name
    env_kwargs = dict(season=args.season, demand_model=world, forecast_model=model_name,
                      elasticity=args.elasticity, capacity=CAPACITY)
    label = MODEL_NAMES[model_name] + (f" (world {MODEL_NAMES[world]})"
                                       if args.common_world else "")
    out_dir = (f"results_{args.season}_{model_name}_e{args.elasticity}"
               + (f"_world-{world}" if args.common_world else ""))
    os.makedirs(out_dir, exist_ok=True)
    log = []

    def out(*lines):
        for line in lines:
            print(line); log.append(line)
        print(); log.append("")

    algo_names = " / ".join(ALGOS[a][2] for a in args.algos)
    env = HotelEnv(**env_kwargs)
    out("=" * 72,
        f" {label} + {algo_names} | STD | season {args.season} | "
        f"{env.dates[0].date()} -> {env.dates[-1].date()} ({env.n_days} days)",
        f" {env.capacity} rooms, elasticity {args.elasticity}, BAR levels "
        f"EUR {env.prices[0]:.0f}-{env.prices[-1]:.0f} (reference: median BAR EUR {env.p_ref:.0f})",
        "=" * 72)

    # 1. Hotel baseline + validation against the real season
    hotel, hotel_path = run_policy(env, hotel_policy, record_path=True)
    a = env.actual
    val = {k: hotel[col].mean() / a[k] - 1 for k, col in
           [("confirmed", "accepted"), ("room_nights", "room_nights"), ("revenue", "revenue")]}
    out("Simulator validation — hotel's prices, simulated vs REAL season:",
        f"  confirmed bookings  {val['confirmed']:+.1%}   room-nights {val['room_nights']:+.1%}"
        f"   revenue {val['revenue']:+.1%}")

    # 2. Best fixed price
    best, _, fixed = find_best_fixed(env)
    out(f"Best fixed price: {PRICE_MULTIPLIERS[best]:.2f}x = BAR EUR {env.prices[best]:.0f}")
    level = 100 * (fixed["reward"].mean() / hotel["reward"].mean() - 1)

    # 3. Each RL algorithm, several seeds, evaluated on the same seasons
    rows = [summarise("Hotel (actual prices)", hotel, hotel, env),
            summarise(f"Best fixed ({PRICE_MULTIPLIERS[best]:.2f}x)", fixed, hotel, env)]
    comp_rows, rl_paths, dynamic = [], {}, {}
    for algo in args.algos:
        name = ALGOS[algo][2]
        seeds, n = args.seeds, len(args.seeds)
        log_dirs = [os.path.join(out_dir, f"logs_{algo}_seed{s}") for s in seeds]
        saves = [os.path.join(out_dir, f"{algo}_seed{s}") for s in seeds]
        jobs = min(args.jobs, n)
        print(f"Training {name} seeds {seeds} ({args.timesteps:,} steps, {jobs} in parallel)...")
        task_args = ([algo] * n, seeds, log_dirs, saves, [env_kwargs] * n, [args.timesteps] * n)
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                outs = list(pool.map(train_and_evaluate, *task_args))
        else:
            outs = list(map(train_and_evaluate, *task_args))
        dfs = [df for df, _ in outs]
        first_path = outs[0][1]
        rows += [summarise(f"{name} seed {s}", df, hotel, env) for s, df in zip(seeds, dfs)]

        mean_df = pd.concat(dfs).groupby(level=0).mean()
        mean_row = summarise(f"{name} mean of {len(args.seeds)} seeds", mean_df, hotel, env)
        rows.append(mean_row)
        rl_paths[name] = first_path
        dynamic[name] = 100 * (mean_df["reward"].mean() / fixed["reward"].mean() - 1)

        plot_learning(log_dirs, args.seeds, name, hotel["reward"].mean(),
                      fixed["reward"].mean(),
                      os.path.join(out_dir, f"learning_curve_{algo}.png"),
                      f"{label} + {name} — learning curve, season {args.season}")

        comp_rows.append({
            "pipeline": f"{label} + {name}",
            "ml_model": label,
            "algo": name,
            "sim_vs_real_revenue_%": 100 * val["revenue"],
            "sim_vs_real_bookings_%": 100 * val["confirmed"],
            "hotel_revenue": hotel["revenue"].mean(),
            "best_fixed_vs_hotel_%": level,
            "rl_revenue": mean_df["revenue"].mean(),
            "rl_vs_hotel_%": mean_row["vs_hotel_%"],
            "rl_vs_hotel_se_%": mean_row["vs_hotel_se_%"],
            "rl_vs_best_fixed_%": dynamic[name],
            "rl_seed_spread_%": 100 * np.std([d["reward"].mean() for d in dfs])
                                / hotel["reward"].mean(),
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    out("Results (same evaluation seasons for all policies; se = paired standard error):",
        table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    lines = ["Decomposition of the RL gain over the hotel:",
             f"  price LEVEL  (best fixed vs hotel)      : {level:+.2f}%   "
             "<- elasticity assumption"]
    for name, dyn in dynamic.items():
        lines.append(f"  DAY-BY-DAY   ({name + ' vs best fixed)':26s}: {dyn:+.2f}%   "
                     "<- value of dynamic pricing")
    out(*lines)
    for name, dyn in dynamic.items():
        if dyn < -0.3:
            out(f"  !!! {name} is below the best fixed price: it has not converged. "
                "Increase --timesteps before drawing conclusions.")

    # 4. Sensitivity (no training): best constant price vs hotel, by elasticity
    if not args.no_sensitivity:
        sens = []
        for eps in SENSITIVITY_ELASTICITIES:
            e = HotelEnv(**{**env_kwargs, "elasticity": eps})
            h, _ = run_policy(e, hotel_policy, n=SENSITIVITY_N_EVAL)
            b, means, _ = find_best_fixed(e, n=SENSITIVITY_N_EVAL)  # same seasons: paired
            sens.append({"elasticity": eps, "best_fixed_multiplier": PRICE_MULTIPLIERS[b],
                         "best_fixed_vs_hotel_%": 100 * (means[b] / h["reward"].mean() - 1)})
        sens = pd.DataFrame(sens)
        sens.to_csv(os.path.join(out_dir, "sensitivity.csv"), index=False)
        out("Sensitivity — best constant price vs hotel, by elasticity:",
            sens.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    plot_price_path(env, hotel_path, rl_paths, os.path.join(out_dir, "price_path.png"),
                    f"{label} + {algo_names} — price path, one simulated season {args.season}")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))

    return comp_rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    ap.add_argument("--common-world", choices=MODELS, default=None)
    ap.add_argument("--elasticity", type=float, default=ELASTICITY)
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--jobs", type=int, default=JOBS,
                    help="parallel processes (1 = sequential); at most one per seed")
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()

    comp = pd.DataFrame([row for m in args.models for row in run_pipeline(m, args)])
    tag = f"{args.season}_e{args.elasticity}" + (
        f"_world-{args.common_world}" if args.common_world else "")
    comp.to_csv(f"comparison_{tag}.csv", index=False)
    plot_comparison(comp, f"comparison_{tag}.png",
                    f"ML model × RL algorithm vs hotel — season {args.season}")

    print("=" * 72)
    print(f" FINAL COMPARISON — season {args.season}, elasticity {args.elasticity}")
    print("=" * 72)
    print(comp.drop(columns=["ml_model", "algo"])
              .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()
    print(" sim_vs_real_*      : how well each simulator reproduces the REAL season")
    print(" rl_vs_hotel_%      : RL revenue vs the hotel's actual pricing, its BAR (same seasons)")
    print(" rl_vs_best_fixed_% : value of day-by-day pricing (must be > 0 to claim it)")
    top = comp.loc[comp["rl_vs_hotel_%"].idxmax()]
    print(f"\n Best combination: {top['pipeline']}  "
          f"{top['rl_vs_hotel_%']:+.2f}% (se {top['rl_vs_hotel_se_%']:.2f}) vs hotel, "
          f"{top['rl_vs_best_fixed_%']:+.2f}% vs best fixed")
    print(f"\n Saved comparison_{tag}.csv / .png")


if __name__ == "__main__":
    main()
