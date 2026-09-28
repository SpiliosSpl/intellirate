import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
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
CAPACITY    = None                 # None -> 17 STD rooms
TIMESTEPS   = 300_000
SEEDS       = [42, 7, 123]
N_EVAL      = 200                  # simulated seasons per policy
EVAL_SEED0  = 100_000              # evaluation seasons: never seen in training

PPO_PARAMS = dict(learning_rate=3e-4, n_steps=2048, batch_size=256,
                  n_epochs=10, gamma=0.99, ent_coef=0.01)

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


def ppo_policy(model):
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
    means = [run_policy(env, fixed_policy(a), n=n)[0]["reward"].mean()
             for a in range(len(PRICE_MULTIPLIERS))]
    return int(np.argmax(means)), means


def train_ppo(seed, log_dir, env_kwargs, timesteps):
    os.makedirs(log_dir, exist_ok=True)
    venv = VecNormalize(DummyVecEnv([lambda: Monitor(HotelEnv(**env_kwargs), log_dir)]),
                        norm_obs=False, norm_reward=True, gamma=PPO_PARAMS["gamma"])
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **PPO_PARAMS)
    model.learn(total_timesteps=timesteps)
    return model


# =============================================================================
# PLOTS
# =============================================================================

def plot_learning(log_dirs, seeds, hotel_mean, fixed_mean, path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for d, sd in zip(log_dirs, seeds):
        x, y = ts2xy(load_results(d), "timesteps")
        w = max(1, len(y) // 30)
        ax.plot(x[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid"), lw=1.8,
                label=f"PPO seed {sd}")
    ax.axhline(hotel_mean, color="#C0392B", ls="--", lw=1.5, label="Hotel (actual prices)")
    ax.axhline(fixed_mean, color="#E67E22", ls=":", lw=1.8, label="Best fixed price")
    ax.set_xlabel("Training timesteps (days simulated)")
    ax.set_ylabel("Season revenue (EUR)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_price_path(env, hotel_path, ppo_path, path, title):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.step(hotel_path["date"], hotel_path["price"], where="mid", color="#C0392B",
             lw=1.4, label="Hotel price (recorded)")
    ax1.step(ppo_path["date"], ppo_path["price"], where="mid", color="#2E75B6",
             lw=1.4, label="PPO price")
    ax1.set_ylabel("Average price per room-night (EUR)")
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
    labels = comp["pipeline"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.18, comp["best_fixed_vs_hotel_%"], 0.36, color="#E67E22",
           label="Best fixed price")
    ax.bar(x + 0.18, comp["ppo_vs_hotel_%"], 0.36, yerr=comp["ppo_vs_hotel_se_%"],
           color="#2E75B6", capsize=4, label="PPO (mean of seeds)")
    ax.axhline(0, color="#C0392B", lw=1.2, ls="--", label="Hotel (actual prices)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Revenue vs hotel (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# =============================================================================
# ONE PIPELINE:  ML model + PPO
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

    env = HotelEnv(**env_kwargs)
    out("=" * 72,
        f" {label} + PPO | STD | season {args.season} | "
        f"{env.dates[0].date()} -> {env.dates[-1].date()} ({env.n_days} days)",
        f" {env.capacity} rooms, elasticity {args.elasticity}, price levels "
        f"EUR {env.prices[0]:.0f}-{env.prices[-1]:.0f} (reference EUR {env.p_ref:.0f})",
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
    best, _ = find_best_fixed(env)
    fixed, _ = run_policy(env, fixed_policy(best))
    out(f"Best fixed price: {PRICE_MULTIPLIERS[best]:.2f}x = EUR {env.prices[best]:.0f}")

    # 3. PPO per seed
    rows = [summarise("Hotel (actual prices)", hotel, hotel, env),
            summarise(f"Best fixed ({PRICE_MULTIPLIERS[best]:.2f}x)", fixed, hotel, env)]
    log_dirs, ppo_dfs, ppo_path = [], [], None
    for seed in args.seeds:
        print(f"Training PPO seed {seed} ({args.timesteps:,} steps)...")
        log_dir = os.path.join(out_dir, f"logs_seed{seed}")
        model = train_ppo(seed, log_dir, env_kwargs, args.timesteps)
        model.save(os.path.join(out_dir, f"ppo_seed{seed}"))
        df, path = run_policy(env, ppo_policy(model), record_path=True)
        rows.append(summarise(f"PPO seed {seed}", df, hotel, env))
        log_dirs.append(log_dir); ppo_dfs.append(df)
        ppo_path = ppo_path if ppo_path is not None else path

    ppo_mean = pd.concat(ppo_dfs).groupby(level=0).mean()
    ppo_row = summarise(f"PPO mean of {len(args.seeds)} seeds", ppo_mean, hotel, env)
    rows.append(ppo_row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    out("Results (same evaluation seasons for all policies; se = paired standard error):",
        table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    level = 100 * (fixed["reward"].mean() / hotel["reward"].mean() - 1)
    dynamic = 100 * (ppo_mean["reward"].mean() / fixed["reward"].mean() - 1)
    out("Decomposition of the PPO gain over the hotel:",
        f"  price LEVEL  (best fixed vs hotel): {level:+.2f}%   <- elasticity assumption",
        f"  DAY-BY-DAY   (PPO vs best fixed)  : {dynamic:+.2f}%   <- value of dynamic pricing")
    if dynamic < -0.3:
        out("  !!! PPO is below the best fixed price: it has not converged. "
            "Increase --timesteps before drawing conclusions.")

    # 4. Sensitivity (no training): best constant price vs hotel, by elasticity
    if not args.no_sensitivity:
        sens = []
        for eps in SENSITIVITY_ELASTICITIES:
            e = HotelEnv(**{**env_kwargs, "elasticity": eps})
            h, _ = run_policy(e, hotel_policy, n=SENSITIVITY_N_EVAL)
            b, means = find_best_fixed(e, n=SENSITIVITY_N_EVAL)     # same seasons: paired
            sens.append({"elasticity": eps, "best_fixed_multiplier": PRICE_MULTIPLIERS[b],
                         "best_fixed_vs_hotel_%": 100 * (means[b] / h["reward"].mean() - 1)})
        sens = pd.DataFrame(sens)
        sens.to_csv(os.path.join(out_dir, "sensitivity.csv"), index=False)
        out("Sensitivity — best constant price vs hotel, by elasticity:",
            sens.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    plot_learning(log_dirs, args.seeds, hotel["reward"].mean(), fixed["reward"].mean(),
                  os.path.join(out_dir, "learning_curve.png"),
                  f"{label} + PPO — learning curve, season {args.season}")
    plot_price_path(env, hotel_path, ppo_path, os.path.join(out_dir, "price_path.png"),
                    f"{label} + PPO — price path, one simulated season {args.season}")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))

    return {
        "pipeline": label,
        "sim_vs_real_revenue_%": 100 * val["revenue"],
        "sim_vs_real_bookings_%": 100 * val["confirmed"],
        "hotel_revenue": hotel["revenue"].mean(),
        "best_fixed_vs_hotel_%": level,
        "ppo_revenue": ppo_mean["revenue"].mean(),
        "ppo_vs_hotel_%": ppo_row["vs_hotel_%"],
        "ppo_vs_hotel_se_%": ppo_row["vs_hotel_se_%"],
        "ppo_vs_best_fixed_%": dynamic,
        "ppo_seed_spread_%": 100 * np.std([d["reward"].mean() for d in ppo_dfs])
                             / hotel["reward"].mean(),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--common-world", choices=MODELS, default=None)
    ap.add_argument("--elasticity", type=float, default=ELASTICITY)
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()

    comp = pd.DataFrame([run_pipeline(m, args) for m in args.models])
    tag = f"{args.season}_e{args.elasticity}" + (
        f"_world-{args.common_world}" if args.common_world else "")
    comp.to_csv(f"comparison_{tag}.csv", index=False)
    plot_comparison(comp, f"comparison_{tag}.png",
                    f"ML1 + PPO vs ML2 + PPO vs hotel — season {args.season}")

    print("=" * 72)
    print(f" FINAL COMPARISON — season {args.season}, elasticity {args.elasticity}")
    print("=" * 72)
    print(comp.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()
    print(" sim_vs_real_*     : how well each simulator reproduces the REAL season")
    print(" ppo_vs_hotel_%    : PPO revenue vs the hotel's actual pricing (same seasons)")
    print(" ppo_vs_best_fixed : value of day-by-day pricing (must be > 0 to claim it)")
    print(f"\n Saved comparison_{tag}.csv / .png")


if __name__ == "__main__":
    main()
