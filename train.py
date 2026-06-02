import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from hotel_env import (
    HotelEnv, BASE_ADR, TOTAL_ROOMS,
    PRICE_MULTIPLIERS, SEASONALITY,
    DAY_MULTIPLIERS, OCCUPANCY_BOOST_WEIGHT,
)

LOG_DIR = "./logs/ppo/" #tensorboard logs for PPO training
DQN_LOG_DIR = "./logs/dqn/" #tensorboard logs for DQN training
TOTAL_TIMESTEPS = 50_000 
N_EVAL_EPISODES = 30

# =========check to perivallon==============
def verify():
    print("=" * 58)
    print("1) --- Environment verification ---")
    print("=" * 58)

    env = HotelEnv()
    check_env(env, warn=True)

    print(f"  !Gymnasium check passed\n")
    print(f"  Total rooms    : {TOTAL_ROOMS}")
    print(f"  Base ADR       : €{BASE_ADR:.2f}")
    print(f"  Price range    : "
          f"€{BASE_ADR * min(PRICE_MULTIPLIERS):.2f} – "
          f"€{BASE_ADR * max(PRICE_MULTIPLIERS):.2f}")
    print(f"  Multipliers    : {PRICE_MULTIPLIERS}")
    print(f"  Seasonality    : {SEASONALITY}")
    print(f"  Day multipliers: {DAY_MULTIPLIERS}")
    print(f"  Occupancy boost: {OCCUPANCY_BOOST_WEIGHT}")
    print(f"  State size     : {env.observation_space.shape[0]} values")
    print(f"  Action space   : {env.action_space.n} discrete levels")

    env.close()
    print()
#================================================


#===========DQN training=============================
def train_dqn():
    print("=" * 58)
    print("2) --- Train DQN ---")
    print("=" * 58)
    print(f"  TensorBoard: tensorboard --logdir {DQN_LOG_DIR}")
    print()

    os.makedirs(DQN_LOG_DIR, exist_ok=True)
    env = Monitor(HotelEnv(), DQN_LOG_DIR)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=DQN_LOG_DIR,
        seed=42,
        device="cpu",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("intellirate_dqn")
    env.close()

    print("\n ! Model saved: intellirate_dqn.zip\n")
    return model
#====================================================


# ==========PPO training=============================
def train_ppo():
    print("=" * 58)
    print("2) --- Train PPO ---")
    print("=" * 58)
    print(f"  Timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Logs      : {LOG_DIR}")
    print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}")
    print()

    os.makedirs(LOG_DIR, exist_ok=True)
    env = Monitor(HotelEnv(), LOG_DIR)

    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        n_steps         = 512, #sb3 default is 2048, but 512 due to small env
        verbose         = 1, # training logs
        tensorboard_log = LOG_DIR,
        seed            = 42,
        device          = "cpu", #change for cuda
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("intellirate_ppo")
    env.close()

    print("\n ! Model saved: intellirate_ppo.zip\n")
    return model
#=============================================================


# ==============RL policy================
def run_policy(policy_fn, n=N_EVAL_EPISODES):
    env = HotelEnv()
    rewards = []
    for ep in range(n):
        obs, _ = env.reset(seed=ep)
        total  = 0.0
        done   = False
        while not done:
            action            = policy_fn(obs)
            obs, r, t, tr, _ = env.step(action)
            total            += r
            done              = t or tr
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))

# ================fixed policy: 5-> 1.00× = 131euro (base ADR)==============
def policy_fixed(obs):
    return 5

# ==================rule-based policy==============
def policy_rule_based(obs):
    occupancy  = (obs[0] + 1.0) / 2.0
    is_weekend = obs[3] > 0.0

    if occupancy > 0.70 or is_weekend:
        return 7    # 1.10× = 144.10
    elif occupancy > 0.35:
        return 5    # 1.00× = 131.00
    else:
        return 2    # 0.85× = 111.35


def compare(model):
    print("=" * 58)
    print("3) --- Policy comparison ---")
    print("=" * 58)

    fixed_m,    fixed_s    = run_policy(policy_fixed) #mean and standard deviation for fixed policy
    rule_m,     rule_s     = run_policy(policy_rule_based) #mean and standard deviation for rule-based policy
    rl_m,      rl_s      = run_policy(
        lambda obs: int(model.predict(obs, deterministic=True)[0])
    )  #mean and standard deviation for rl policy

    #Improvement over fixed-price baseline (%)
    rule_imp = (rule_m - fixed_m) / fixed_m * 100
    rl_imp  = (rl_m  - fixed_m) / fixed_m * 100

    results = {
        "Fixed price (1.00x)": (fixed_m, fixed_s, 0.0),
        "Rule-based": (rule_m,  rule_s,  rule_imp),
        "RL model (DQN / PPO)": (rl_m,   rl_s,   rl_imp),
    }

    # Print results table
    print(f"  {'Policy':<24} {'Mean reward':>12} "
          f"{'Std':>8} {'vs Fixed':>10}")
    print("  " + "─" * 56)
    for name, (m, s, imp) in results.items():
        imp_str = f"{imp:+.1f}%" if imp != 0.0 else "—"
        print(f"  {name:<24} {m:>12.2f} {s:>8.2f} {imp_str:>10}")

    return results


# =========Curve plotting=============================
def plot(results, path="learning_curve.png"):
    print("=" * 58)
    print("4) --- Plotting curve ---")
    print("=" * 58)

    try:
        x, y = ts2xy(load_results(LOG_DIR), "timesteps")
        if len(y) == 0:
            print("!No training data to plot yet!")
            return

        # Smooth the noisy per-episode reward curve
        window = max(1, len(y) // 20)
        y_sm   = np.convolve(y, np.ones(window) / window, mode="valid")
        x_sm   = x[window - 1:]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Raw faint color + smoothed rl curve
        ax.plot(x,    y,    alpha=0.15, color="#2E75B6", linewidth=0.8)
        ax.plot(x_sm, y_sm, color="#2E75B6", linewidth=2.2, label="RL model (DQN / PPO)")

        # Baseline horizontal reference lines
        fixed_m = results["Fixed price (1.00x)"][0]
        rule_m  = results["Rule-based"][0]
        ax.axhline(
            fixed_m, color="#E24B4A", linestyle="--", linewidth=1.5,
            label=f"Fixed price  (mean {fixed_m:.1f})"
        )
        ax.axhline(
            rule_m, color="#EF9F27", linestyle="--", linewidth=1.5,
            label=f"Rule-based  (mean {rule_m:.1f})"
        )

        ax.set_xlabel("Training Timesteps", fontsize=12)
        ax.set_ylabel("Episode Reward  (RevPAR)", fontsize=12)
        ax.set_title(
            "RL Learning Curve vs Baselines\n"
            f"1 room type - "
            f"Base ADR €{BASE_ADR:.0f} - "
            f"Multipliers {PRICE_MULTIPLIERS[0]}x–{PRICE_MULTIPLIERS[-1]}x",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150) #save the plot in the repo
        print(f"! Plot saved: {path}")
        plt.show() # display the plot once saved

    except Exception as e:
        print(f"!Could not plot: {e} !")
        print(f"Run: tensorboard --logdir {LOG_DIR}")


# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 58)
    print("  IntelliRate — RL")
    print("  1 room type - simulated hotel environment")
    print(f"  Base ADR: {BASE_ADR:.2f}  -  "
          f"10 price levels")
    print("=" * 58)
    print()

    verify()
    #model = train_ppo() #runs with PPO
    model = train_dqn()  #runs with DQN
    results = compare(model)
    plot(results)