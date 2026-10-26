import argparse
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import joblib
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.results_plotter import load_results, ts2xy

from hotel_season_env import HotelEnv, PRICE_MULTIPLIERS, SIM_DATA_PATH


# =============================================================================
# CONFIGURATION
# =============================================================================

SEASON      = 2026
MODELS      = ["rf", "xgb"]
MODEL_NAMES = {"rf": "ML1 RF", "xgb": "ML2 XGB"}
ELASTICITY  = 1.0
TIMESTEPS   = 900_000              # per algorithm and seed
SEEDS       = [42, 7, 123, 1, 2024]
JOBS        = 5                    # parallel processes: the seeds of one ML x RL train together

# Simulated seasons by role, told apart by their seed. Training uses each training
# seed's own SB3 stream. Validation chooses (checkpoints, best fixed price); test is
# only reported, never used to choose anything, and looked at once.
VAL_SEED0   = 100_000              # the old evaluation seasons: past design choices used them
N_VAL       = 50                   # checkpoints: the first N_VAL validation seasons
N_VAL_FIXED = 200                  # best fixed price: no training, so cheap -> more seasons
TEST_SEED0  = 500_000
N_TEST      = 200
EVAL_EVERY  = 25_000               # training steps between validations of the greedy policy
N_VAL_SCHEDULE = 1000              # per-date schedule: one choice per date -> many seasons
ENSEMBLE_SEED0 = VAL_SEED0 + N_VAL # seed ensemble vs mean of seeds: seasons no checkpoint saw
N_ENSEMBLE_VAL = 200
assert max(VAL_SEED0 + max(N_VAL, N_VAL_FIXED, N_VAL_SCHEDULE),
           ENSEMBLE_SEED0 + N_ENSEMBLE_VAL) <= TEST_SEED0, "validation and test seasons overlap"

# Low gamma / GAE lambda: today's price barely affects later days (mean stay 1.26
# nights), so a long horizon only adds future demand noise to each advantage.
# gamma, gae_lambda and the Double DQN changes below were chosen on a small
# validation grid (300k steps, rf world, 3 seeds; see CLAUDE.md).
PPO_PARAMS = dict(learning_rate=3e-4, n_steps=2048, batch_size=256,
                  n_epochs=10, gamma=0.9, gae_lambda=0.5, ent_coef=0.01)

# Double DQN: every setting is written out (not left to SB3 defaults). Batch 512,
# gamma 0.9, a soft target update every gradient step and a decaying learning
# rate were tested together, not one by one. Same network size as PPO.
DQN_PARAMS = dict(learning_rate=lambda p: 1e-4 * p,     # p: progress remaining, 1 -> 0
                  buffer_size=100_000, learning_starts=5_000,
                  batch_size=512, gamma=0.9, train_freq=4, gradient_steps=1,
                  target_update_interval=4, tau=0.005,
                  exploration_fraction=0.3, exploration_initial_eps=1.0,
                  exploration_final_eps=0.02, max_grad_norm=10,
                  policy_kwargs=dict(net_arch=[64, 64]))



class DoubleDQN(DQN):
    def next_q_values(self, next_obs):
        next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
        return torch.gather(self.q_net_target(next_obs), dim=1, index=next_actions)

    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with torch.no_grad():
                next_q = self.next_q_values(replay_data.next_observations)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1,
                                            index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# RL algorithms: key -> (SB3 class, hyperparameters, display name).
ALGOS = {
    "ppo": (PPO, PPO_PARAMS, "PPO"),
    "ddqn": (DoubleDQN, DQN_PARAMS, "Double DQN"),
}
# Training reward per algorithm: True = revenue minus the best fixed price's
# dry-run revenue on the same day (BaselineReward). Validation and test use revenue.
# Chosen on holdout seasons (300k steps, rf world, 5 tuning seeds; see CLAUDE.md).
BASELINE_REWARD = {"ppo": True, "ddqn": True}
RL_COLOURS = ["#2E75B6", "#8E44AD", "#16A085", "#7F8C8D"]   # by position in --algos

SENSITIVITY_ELASTICITIES = [0.6, 0.8, 1.0, 1.2, 1.5]
SENSITIVITY_N_EVAL       = N_TEST  # all test seasons: the eps = 1.0 row equals the main result

ELASTICITY_LOG = "elasticity_log_{model}.csv"   # one row appended per run
PROTOCOL       = "val-test"                     # logged: rows of older runs used one season set
LOG_FIXED      = ["timestamp", "protocol", "sim_data", "season", "elasticity", "timesteps",
                  "seeds", "hotel_revenue", "best_fixed_multiplier", "best_fixed_revenue"]


# =============================================================================
# POLICY EVALUATION
# =============================================================================

def new_season():
    return dict(revenue=0.0, reward=0.0, room_nights=0, accepted=0, cancelled=0, lost=0,
                above_range=0)


def add_day(tot, reward, info, top_price):
    """Add one day to a season's totals. above_range counts the days priced
    above the hotel's highest price (top_price), where g(p) is extrapolated."""
    tot["reward"] += reward
    for key in ("revenue", "room_nights", "accepted", "cancelled", "lost"):
        tot[key] += info[key]
    tot["above_range"] += info["price"] > top_price


def run_policy(env, act, n=N_TEST, seed0=TEST_SEED0, record_path=False):
    """act(env, obs) -> result of env.step / env.step_price. Default: the test seasons."""
    rows, path, top = [], None, env.hotel_price.max()
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        tot, trace, done = new_season(), [], False
        while not done:
            obs, reward, done, _, info = act(env, obs)
            add_day(tot, reward, info, top)
            if record_path and i == 0:
                trace.append((info["date"], info["price"], info["occupied"]))
        rows.append(tot)
        if record_path and i == 0:
            path = pd.DataFrame(trace, columns=["date", "price", "occupied"])
    return pd.DataFrame(rows), path


def make_envs(env_kwargs, n):
    """n copies of the env, one per season run in lockstep, sharing one load of the sim data."""
    sim = joblib.load(SIM_DATA_PATH)
    return [HotelEnv(**env_kwargs, sim=sim) for _ in range(n)]


def run_batched(envs, predict, seed0, record_path=False):
    """run_policy for agents: the len(envs) seasons from seed0 in lockstep, with
    one batched predict(observations) -> actions per day. Several times faster
    than one season at a time, with the same actions up to float rounding: a
    batched matrix product can flip a near-exact tie between two actions."""
    obs = np.stack([env.reset(seed=seed0 + i)[0] for i, env in enumerate(envs)])
    tots, trace, top = [new_season() for _ in envs], [], envs[0].hotel_price.max()
    for _ in range(envs[0].n_days):
        steps = [env.step(int(a)) for env, a in zip(envs, predict(obs))]
        obs = np.stack([s[0] for s in steps])
        for tot, (_, reward, _, _, info) in zip(tots, steps):
            add_day(tot, reward, info, top)
        if record_path:
            info = steps[0][4]
            trace.append((info["date"], info["price"], info["occupied"]))
    path = pd.DataFrame(trace, columns=["date", "price", "occupied"]) if record_path else None
    return pd.DataFrame(tots), path


def hotel_policy(env, obs):
    return env.step_price(env.hotel_price[env.t])


def fixed_policy(action):
    return lambda env, obs: env.step(action)


def schedule_policy(schedule):
    return lambda env, obs: env.step(int(schedule[env.t]))


def greedy(model):
    """Batched greedy actions of a trained SB3 agent (PPO: mode, Double DQN: argmax Q)."""
    return lambda obs: model.predict(obs, deterministic=True)[0]


def ensemble(models):
    """Batched actions of an odd number of agents combined: per season, the
    median of their greedy actions (price levels are ordered, so a median, not
    a vote; with an even number the median would favour one side)."""
    def predict(obs):
        actions = np.sort([m.predict(obs, deterministic=True)[0] for m in models], axis=0)
        return actions[len(models) // 2]
    return predict


def summarise(name, df, base, env):
    diff = df["reward"].values - base["reward"].values          # paired
    avail = env.online_avail.sum()                              # online room-nights for sale
    se = 100 * diff.std(ddof=1) / np.sqrt(len(diff)) / base["reward"].mean()
    return {
        "policy": name,
        "revenue": df["revenue"].mean(),
        "vs_hotel_%": 100 * diff.mean() / base["reward"].mean(),
        "vs_hotel_se_%": se,
        "vs_hotel_ci95_%": stats.norm.ppf(0.975) * se,
        "room_nights": df["room_nights"].mean(),
        "occupancy_%": 100 * df["room_nights"].mean() / avail,
        "ADR": df["revenue"].mean() / df["room_nights"].mean(),
        "RevPAR": df["revenue"].mean() / avail,
        "lost_requests": df["lost"].mean(),
        "above_range_%": 100 * df["above_range"].mean() / env.n_days,
    }


def seed_se(dfs, base):
    """SE and 95% CI half-width (% of base) of the mean-of-seeds gain over base:
    the variance between training seeds plus the season noise of the
    seed-averaged run. With one seed: season noise only, CI nan."""
    n, ref = len(dfs), base["reward"].mean()
    diff = pd.concat(dfs).groupby(level=0).mean()["reward"].values - base["reward"].values
    season = 100 * diff.std(ddof=1) / np.sqrt(len(diff)) / ref
    if n < 2:
        return season, np.nan
    gains = [100 * (df["reward"].values - base["reward"].values).mean() / ref for df in dfs]
    se = np.sqrt(np.var(gains, ddof=1) / n + season ** 2)
    return se, stats.t.ppf(0.975, n - 1) * se


def find_best_fixed(env, n=N_VAL_FIXED, seed0=VAL_SEED0, actions=None):
    """Best constant price level (among actions, default all), chosen on the
    validation seasons: (its action, its run on those seasons)."""
    actions = list(range(len(PRICE_MULTIPLIERS))) if actions is None else list(actions)
    runs = [run_policy(env, fixed_policy(a), n=n, seed0=seed0)[0] for a in actions]
    best = int(np.argmax([df["reward"].mean() for df in runs]))
    return actions[best], runs[best]


def action_sweep(env, n=N_VAL_SCHEDULE, seed0=VAL_SEED0):
    """Daily reward of every price level held all season, on n validation
    seasons: an array (levels, seasons, days)."""
    out = np.empty((len(PRICE_MULTIPLIERS), n, env.n_days))
    for a in range(len(PRICE_MULTIPLIERS)):
        for i in range(n):
            env.reset(seed=seed0 + i)
            for t in range(env.n_days):
                out[a, i, t] = env.step(a)[1]
    return out


def best_schedule(sweep, actions):
    """Per date, the level among actions with the best mean daily reward in the
    sweep: a per-date (open-loop) schedule. A benchmark, not a ceiling: each
    level was swept for a whole season, so it ignores that one night's sales
    change the rooms left on the next, and a policy that sees them can beat it."""
    actions = np.asarray(actions)
    return actions[sweep[actions].mean(axis=1).argmax(axis=0)]


def observed_range(env):
    """The price levels inside the range of the hotel's own prices, where the
    price response g(p) is not extrapolated beyond the data."""
    lo, hi = env.hotel_price.min() / env.p_ref, env.hotel_price.max() / env.p_ref
    return [a for a, m in enumerate(PRICE_MULTIPLIERS) if lo <= m <= hi]


def at_grid_edge(action):
    """A best price at an end of PRICE_MULTIPLIERS: the optimum may lie beyond it."""
    return action in (0, len(PRICE_MULTIPLIERS) - 1)


def log_columns():
    """The elasticity log's header: fixed fields, then two per algorithm in ALGOS."""
    return LOG_FIXED + [f"{algo.upper()}_{col}" for algo in ALGOS
                        for col in ("mean_revenue", "mean_occupancy_%")]


def elasticity_log_path(model_name, args):
    world = f"_world-{args.common_world}" if args.common_world else ""
    return ELASTICITY_LOG.format(model=model_name + world)


def steps_tag(timesteps):
    """Timesteps in output names: 900000 -> 900k, 12345 -> 12345."""
    return f"{timesteps // 1000}k" if timesteps % 1000 == 0 else str(timesteps)


def check_elasticity_log(path):
    """Fail if an existing log has another header (main checks before training)."""
    if os.path.exists(path) and pd.read_csv(path, nrows=0).columns.tolist() != log_columns():
        raise ValueError(f"{path} has different columns (ALGOS or the log changed?): "
                         "rename or delete it")


def append_elasticity_log(model_name, args, table, best_multiplier):
    """Append this run's hotel and best fixed revenue and the mean RL revenue
    and occupancy to the model's log, to track how they change with the
    elasticity across runs. sim_data identifies the world: rows with the same
    one come from the same sim_data.pkl (every ML rerun gives a new one)."""
    from ml_demand_model import file_hash         # here: workers don't load the ML stack
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "protocol": PROTOCOL,
           "sim_data": file_hash(SIM_DATA_PATH)[:12],
           "season": args.season, "elasticity": args.elasticity,
           "timesteps": args.timesteps, "seeds": " ".join(map(str, args.seeds)),
           "hotel_revenue": table["revenue"].iloc[0],           # rows 0-1: hotel, best fixed
           "best_fixed_multiplier": best_multiplier,
           "best_fixed_revenue": table["revenue"].iloc[1]}
    by_policy = table.set_index("policy")
    for algo, (_, _, name) in ALGOS.items():      # every algorithm: a fixed header
        mean = f"{name} mean of {len(args.seeds)} seeds"
        trained = mean in by_policy.index
        row[f"{algo.upper()}_mean_revenue"] = by_policy.loc[mean, "revenue"] if trained else np.nan
        row[f"{algo.upper()}_mean_occupancy_%"] = by_policy.loc[mean, "occupancy_%"] if trained else np.nan
    path = elasticity_log_path(model_name, args)
    check_elasticity_log(path)
    pd.DataFrame([row])[log_columns()].to_csv(path, mode="a", header=not os.path.exists(path),
                                               index=False)
    print(f"Appended to {path}")


class ValidationCheckpoint(BaseCallback):
    """Greedy policy on the validation seasons every `every` steps and at the end
    of training; saves the best model to save_path. It has its own envs with
    seeded resets, and a deterministic predict draws no random numbers, so the
    training run is the same as without it."""

    def __init__(self, env_kwargs, save_path, every):
        super().__init__()
        self.val_envs = make_envs(env_kwargs, N_VAL)
        self.save_path, self.every = save_path, every
        self.curve, self.best, self.best_step = [], -np.inf, None

    def _validate(self):
        df, _ = run_batched(self.val_envs, greedy(self.model), VAL_SEED0)
        score = df["reward"].mean()
        self.curve.append((self.num_timesteps, score))
        if score > self.best:                       # ties keep the earlier checkpoint
            self.best, self.best_step = score, self.num_timesteps
            self.model.save(self.save_path)

    def _on_step(self):
        if self.num_timesteps % self.every == 0:    # one env: one step per call
            self._validate()
        return True

    def _on_training_end(self):
        if not self.curve or self.curve[-1][0] != self.num_timesteps:
            self._validate()


class BaselineReward(gym.Wrapper):
    """Training reward: revenue minus what the baseline price (the best fixed
    price) would have earned on the same day from the same pre-drawn requests
    and calendar (HotelEnv.dry_run). The demand noise both share cancels, and
    the subtracted term doesn't depend on today's action; only multi-night
    stays add a small bias, through the calendar. Validation and test use
    plain revenue."""

    def __init__(self, env, price):
        super().__init__(env)
        self.price = price

    def step(self, action):
        base = self.env.unwrapped.dry_run(self.price)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - base, terminated, truncated, info


def train_agent(algo, seed, log_dir, save_path, env_kwargs, timesteps, eval_every, baseline=None):
    """Train, then reload the checkpoint with the best validation score.
    baseline: the price level (action) whose dry-run reward is subtracted from
    the training reward (BaselineReward); None trains on revenue."""
    cls, params, _ = ALGOS[algo]
    os.makedirs(log_dir, exist_ok=True)

    def make_env():
        env = Monitor(HotelEnv(**env_kwargs), log_dir)      # logs the true revenue
        return env if baseline is None else BaselineReward(env, env.unwrapped.prices[baseline])

    # Reward normalisation: off-policy agents store raw rewards in the replay
    # buffer and normalise them when sampled, so this is valid for Double DQN too.
    venv = VecNormalize(DummyVecEnv([make_env]),
                        norm_obs=False, norm_reward=True, gamma=params["gamma"])
    model = cls("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **params)
    val = ValidationCheckpoint(env_kwargs, save_path, eval_every)
    model.learn(total_timesteps=timesteps, callback=val)
    venv.close()                                    # closes the Monitor's monitor.csv
    return cls.load(save_path, device="cpu"), val


def train_and_evaluate(algo, seed, log_dir, save_path, env_kwargs, timesteps, eval_every,
                       baseline=None):
    """One seed: train, keep the best validation checkpoint, test it. Top-level
    so that worker processes (spawned on Windows) can import it. One torch
    thread per process: the network is too small to gain from more, and it
    measured faster."""
    torch.set_num_threads(1)
    model, val = train_agent(algo, seed, log_dir, save_path, env_kwargs, timesteps, eval_every,
                             baseline)
    df, path = run_batched(make_envs(env_kwargs, N_TEST), greedy(model), TEST_SEED0,
                           record_path=True)
    return df, path, val.curve, val.best_step


# =============================================================================
# PLOTS
# =============================================================================

def plot_learning(log_dirs, curves, seeds, algo_name, hotel_val, fixed_val, path, title):
    """Top: training episodes (exploration included). Bottom: greedy policy on the
    validation seasons, with the selected checkpoint starred."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for d, curve, sd in zip(log_dirs, curves, seeds):
        x, y = ts2xy(load_results(d), "timesteps")
        w = max(1, len(y) // 30)
        ax1.plot(x[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid"), lw=1.8,
                 label=f"{algo_name} seed {sd}")
        steps, scores = zip(*curve)
        line, = ax2.plot(steps, scores, marker="o", ms=3, lw=1.5, label=f"{algo_name} seed {sd}")
        i = int(np.argmax(scores))
        ax2.plot(steps[i], scores[i], marker="*", ms=14, color=line.get_color())
    for ax in (ax1, ax2):
        ax.axhline(hotel_val, color="#C0392B", ls="--", lw=1.5, label="Hotel (actual prices)")
        ax.axhline(fixed_val, color="#E67E22", ls=":", lw=1.8, label="Best fixed price")
        ax.set_ylabel("Season revenue (EUR)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax1.set_title(f"{title}\ntraining episodes (exploration included)")
    ax2.set_title(f"greedy policy on {N_VAL} validation seasons (star: selected checkpoint)")
    ax2.set_xlabel("Training timesteps (days simulated)")
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
    ax.set_ylabel("Revenue vs hotel (%), test seasons")
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
                      elasticity=args.elasticity)
    label = MODEL_NAMES[model_name] + (f" (world {MODEL_NAMES[world]})"
                                       if args.common_world else "")
    suffix = f"_world-{world}" if args.common_world else ""
    out_dir = (f"results_{args.season}_{model_name}_e{args.elasticity}"
               f"_t{steps_tag(args.timesteps)}{suffix}")
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
        f" {env.capacity} rooms ({env.online_avail.mean():.1f} online per night), "
        f"elasticity {args.elasticity}, BAR levels "
        f"EUR {env.prices[0]:.0f}-{env.prices[-1]:.0f} (reference: median BAR EUR {env.p_ref:.0f})",
        f" Seasons: training = each seed's own stream; validation = seeds "
        f"{VAL_SEED0}-{VAL_SEED0 + N_VAL - 1} (checkpoints), "
        f"{VAL_SEED0}-{VAL_SEED0 + N_VAL_FIXED - 1} (best fixed price),",
        f" {VAL_SEED0}-{VAL_SEED0 + N_VAL_SCHEDULE - 1} (per-date schedule), "
        f"{ENSEMBLE_SEED0}-{ENSEMBLE_SEED0 + N_ENSEMBLE_VAL - 1} (seed ensemble); "
        f"test = seeds {TEST_SEED0}-{TEST_SEED0 + N_TEST - 1} (every reported number)",
        " Training reward: " + ", ".join(
            f"{ALGOS[a][2]} {'revenue - best fixed dry run' if BASELINE_REWARD[a] else 'revenue'}"
            for a in args.algos),
        "=" * 72)

    # 1. Hotel baseline (test, validation) + validation of the simulator against the real season
    hotel, hotel_path = run_policy(env, hotel_policy, record_path=True)
    hotel_val, _ = run_policy(env, hotel_policy, n=N_VAL_FIXED, seed0=VAL_SEED0)
    hotel_ckpt = hotel_val["reward"].iloc[:N_VAL].mean()     # on the checkpoint seasons
    a = env.actual
    val = {k: hotel[col].mean() / a[k] - 1 for k, col in
           [("confirmed", "accepted"), ("room_nights", "room_nights"), ("revenue", "revenue")]}
    out("Simulator validation — hotel's prices, simulated vs REAL season:",
        f"  confirmed bookings  {val['confirmed']:+.1%}   room-nights {val['room_nights']:+.1%}"
        f"   revenue {val['revenue']:+.1%}")

    # 2. Best fixed price: chosen on validation, reported on test
    best, fixed_val = find_best_fixed(env)
    fixed, _ = run_policy(env, fixed_policy(best))
    out(f"Best fixed price (chosen on {N_VAL_FIXED} validation seasons): "
        f"{PRICE_MULTIPLIERS[best]:.2f}x = BAR EUR {env.prices[best]:.0f}")
    if at_grid_edge(best):
        out("  !!! The best fixed price is at the edge of the price grid: the best level may "
            "lie beyond it, and the RL agents are capped there too")
    level = 100 * (fixed["reward"].mean() / hotel["reward"].mean() - 1)
    fixed_val_gain = 100 * (fixed_val["reward"].mean() / hotel_val["reward"].mean() - 1)

    # 2b. Benchmarks (no training), chosen on validation, reported on test: the best
    # price per date, and the best fixed price and per-date schedule within the
    # hotel's own price range, where the price response is not extrapolated
    sweep, in_range = action_sweep(env), observed_range(env)
    per_date, _ = run_policy(env, schedule_policy(best_schedule(sweep, range(len(PRICE_MULTIPLIERS)))))
    per_date_in, _ = run_policy(env, schedule_policy(best_schedule(sweep, in_range)))
    del sweep
    best_in, _ = find_best_fixed(env, actions=in_range)
    fixed_in, _ = run_policy(env, fixed_policy(best_in))
    schedule_gain = 100 * (per_date["reward"].mean() / fixed["reward"].mean() - 1)
    level_in = 100 * (fixed_in["reward"].mean() / hotel["reward"].mean() - 1)
    out(f"Hotel's own price range: {env.hotel_price.min() / env.p_ref:.2f}-"
        f"{env.hotel_price.max() / env.p_ref:.2f}x the reference -> levels "
        f"{PRICE_MULTIPLIERS[in_range[0]]:.2f}-{PRICE_MULTIPLIERS[in_range[-1]]:.2f}x; best fixed "
        f"there {PRICE_MULTIPLIERS[best_in]:.2f}x. Per-date schedules chosen on "
        f"{N_VAL_SCHEDULE} validation seasons.")

    # 3. Each RL algorithm, several seeds, evaluated on the same test seasons
    rows = [summarise("Hotel (actual prices)", hotel, hotel, env),
            {**summarise(f"Best fixed ({PRICE_MULTIPLIERS[best]:.2f}x)", fixed, hotel, env),
             "val_vs_hotel_%": fixed_val_gain},
            summarise("Per-date schedule (benchmark)", per_date, hotel, env),
            summarise(f"Best fixed in hotel's range ({PRICE_MULTIPLIERS[best_in]:.2f}x)",
                      fixed_in, hotel, env),
            summarise("Per-date schedule in hotel's range", per_date_in, hotel, env)]
    with_ensemble = len(args.seeds) >= 3 and len(args.seeds) % 2 == 1
    if with_ensemble:                         # the seed ensemble's validation reference
        hotel_ens = run_policy(env, hotel_policy, n=N_ENSEMBLE_VAL,
                               seed0=ENSEMBLE_SEED0)[0]["reward"].mean()
    comp_rows, rl_paths, dynamic, vs_hotel, curves_out = [], {}, {}, {}, []
    for algo in args.algos:
        name = ALGOS[algo][2]
        seeds, n = args.seeds, len(args.seeds)
        log_dirs = [os.path.join(out_dir, f"logs_{algo}_seed{s}") for s in seeds]
        saves = [os.path.join(out_dir, f"{algo}_seed{s}") for s in seeds]
        jobs = min(args.jobs, n)
        print(f"Training {name} seeds {seeds} ({args.timesteps:,} steps, {jobs} in parallel)...")
        baseline = best if BASELINE_REWARD[algo] else None
        task_args = ([algo] * n, seeds, log_dirs, saves, [env_kwargs] * n,
                     [args.timesteps] * n, [args.eval_every] * n, [baseline] * n)
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                outs = list(pool.map(train_and_evaluate, *task_args))
        else:
            outs = list(map(train_and_evaluate, *task_args))
        dfs = [o[0] for o in outs]
        curves = [o[2] for o in outs]
        first_path = outs[0][1]
        for s, df, curve, step in zip(seeds, dfs, curves, [o[3] for o in outs]):
            best_val = max(score for _, score in curve)
            rows.append({**summarise(f"{name} seed {s}", df, hotel, env),
                         "checkpoint_step": step, "val_reward": best_val,
                         "val_vs_hotel_%": 100 * (best_val / hotel_ckpt - 1)})
            curves_out += [{"algo": algo, "seed": s, "timesteps": t, "val_reward": r}
                           for t, r in curve]

        mean_df = pd.concat(dfs).groupby(level=0).mean()
        mean_row = summarise(f"{name} mean of {n} seeds", mean_df, hotel, env)
        mean_row["vs_hotel_se_%"], mean_row["vs_hotel_ci95_%"] = seed_se(dfs, hotel)
        rows.append(mean_row)
        vs_hotel[name] = mean_row["vs_hotel_%"]
        rl_paths[name] = first_path
        dynamic[name] = 100 * (mean_df["reward"].mean() / fixed["reward"].mean() - 1)
        dyn_se, dyn_ci = seed_se(dfs, fixed)
        if n < 2:
            out(f"  !!! {name}: one seed, so the error bars hold season noise only")
        ens_gain, ens_better = np.nan, np.nan
        if not with_ensemble:
            out(f"  {name}: no seed ensemble (it needs an odd number of seeds, at least 3)")
        else:
            # Seed ensemble: every seed's checkpoint, the median action each day.
            # Compared with the mean of seeds on validation seasons no checkpoint saw.
            models = [ALGOS[algo][0].load(s, device="cpu") for s in saves]
            ens, _ = run_batched(make_envs(env_kwargs, N_TEST), ensemble(models), TEST_SEED0)
            val_envs = make_envs(env_kwargs, N_ENSEMBLE_VAL)
            ens_val = run_batched(val_envs, ensemble(models), ENSEMBLE_SEED0)[0]["reward"].mean()
            seeds_val = np.mean([run_batched(val_envs, greedy(m), ENSEMBLE_SEED0)[0]["reward"].mean()
                                 for m in models])
            ens_better = bool(ens_val > seeds_val)
            ens_gain = 100 * (ens["reward"].mean() / fixed["reward"].mean() - 1)
            rows.append({**summarise(f"{name} ensemble (median of {n} seeds)", ens, hotel, env),
                         "val_vs_hotel_%": 100 * (ens_val / hotel_ens - 1)})
            out(f"  {name} seed ensemble on validation seasons {ENSEMBLE_SEED0}-"
                f"{ENSEMBLE_SEED0 + N_ENSEMBLE_VAL - 1}: {100 * (ens_val / hotel_ens - 1):+.2f}% vs "
                f"hotel, mean of seeds {100 * (seeds_val / hotel_ens - 1):+.2f}% -> "
                + ("the ensemble is better" if ens_better else "the mean of seeds is better"))

        plot_learning(log_dirs, curves, seeds, name, hotel_ckpt,
                      fixed_val["reward"].iloc[:N_VAL].mean(),
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
            "rl_vs_hotel_ci95_%": mean_row["vs_hotel_ci95_%"],
            "rl_vs_best_fixed_%": dynamic[name],
            "rl_vs_best_fixed_se_%": dyn_se,
            "rl_vs_best_fixed_ci95_%": dyn_ci,
            "rl_seed_spread_%": (100 * np.std([d["reward"].mean() for d in dfs], ddof=1)
                                 / hotel["reward"].mean() if n > 1 else np.nan),
            "schedule_vs_best_fixed_%": schedule_gain,
            "best_fixed_in_range_vs_hotel_%": level_in,
            "ensemble_vs_best_fixed_%": ens_gain,
            "ensemble_beats_seeds_on_val": ens_better,
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    pd.DataFrame(curves_out).to_csv(os.path.join(out_dir, "validation_curve.csv"), index=False)
    out("Results on the TEST seasons (val_* columns: validation seasons; se = paired, "
        "mean-of-seeds rows include the variance between seeds):",
        table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    lines = ["Decomposition of the RL gain over the hotel (test seasons):",
             f"  price LEVEL  (best fixed vs hotel)      : {level:+.2f}%   "
             "<- elasticity assumption",
             f"  LEVEL in the hotel's own price range    : {level_in:+.2f}%   "
             f"<- best fixed {PRICE_MULTIPLIERS[best_in]:.2f}x: no price beyond the data",
             f"  per-date SCHEDULE vs best fixed         : {schedule_gain:+.2f}%   "
             "<- day-by-day value of a per-date schedule (benchmark)"]
    for name, gain in vs_hotel.items():
        lines.append(f"  RL vs HOTEL  ({name + ' vs hotel)':26s}: {gain:+.2f}%   "
                     "<- level + day-by-day")
    out(*lines)
    for name, dyn in dynamic.items():
        if dyn < -0.3:
            out(f"  !!! {name} is below the best fixed price: it has not converged. "
                "Increase --timesteps before drawing conclusions.")

    # 4. Sensitivity (no training): best constant price chosen on validation, vs hotel on test
    if not args.no_sensitivity:
        sens, edges = [], []
        for eps in SENSITIVITY_ELASTICITIES:
            e = HotelEnv(**{**env_kwargs, "elasticity": eps})
            b, _ = find_best_fixed(e)
            h, _ = run_policy(e, hotel_policy, n=SENSITIVITY_N_EVAL)       # same seasons: paired
            f, _ = run_policy(e, fixed_policy(b), n=SENSITIVITY_N_EVAL)
            sens.append({"elasticity": eps, "best_fixed_multiplier": PRICE_MULTIPLIERS[b],
                         "best_fixed_vs_hotel_%": 100 * (f["reward"].mean() / h["reward"].mean() - 1)})
            if at_grid_edge(b):
                edges.append(f"  !!! Elasticity {eps}: the best fixed price is at the edge of the "
                             f"price grid ({PRICE_MULTIPLIERS[b]:.2f}x), so its gain is capped by the grid")
        sens = pd.DataFrame(sens)
        sens.to_csv(os.path.join(out_dir, "sensitivity.csv"), index=False)
        out("Sensitivity — best constant price (chosen on validation) vs hotel on test, by elasticity:",
            sens.to_string(index=False, float_format=lambda v: f"{v:.2f}"), *edges)

    plot_price_path(env, hotel_path, rl_paths, os.path.join(out_dir, "price_path.png"),
                    f"{label} + {algo_names} — price path, one simulated season {args.season}")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log))
    append_elasticity_log(model_name, args, table, PRICE_MULTIPLIERS[best])

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
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                    help="training steps between validations of the greedy policy")
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()
    for m in args.models:                 # fail now, not after hours of training
        check_elasticity_log(elasticity_log_path(m, args))

    comp = pd.DataFrame([row for m in args.models for row in run_pipeline(m, args)])
    tag = f"{args.season}_e{args.elasticity}_t{steps_tag(args.timesteps)}" + (
        f"_world-{args.common_world}" if args.common_world else "")
    comp.to_csv(f"comparison_{tag}.csv", index=False)
    plot_comparison(comp, f"comparison_{tag}.png",
                    f"ML model × RL algorithm vs hotel — season {args.season}")

    print("=" * 72)
    print(f" FINAL COMPARISON — season {args.season}, elasticity {args.elasticity} (test seasons)")
    print("=" * 72)
    print(comp.drop(columns=["ml_model", "algo"])
              .to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print()
    print(" sim_vs_real_*      : how well each simulator reproduces the REAL season")
    print(" rl_vs_hotel_%      : RL revenue vs the hotel's actual pricing, its BAR (same seasons)")
    print(" rl_vs_best_fixed_% : value of day-by-day pricing (must be > 0 to claim it)")
    print(" *_se_% / *_ci95_%  : include the variance between training seeds")
    print(f"\n Saved comparison_{tag}.csv / .png")


if __name__ == "__main__":
    main()
