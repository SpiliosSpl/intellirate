"""
IntelliRate — Hotel Season Simulator (Gymnasium environment)
=============================================================
One episode = one operating season of the hotel (2025 or 2026).
One step    = one arrival date. The agent sets that date's price level.

TWO ML MODELS (from ml_demand_model.py): "rf" (ML1) and "xgb" (ML2)
    demand_model   : defines the simulated WORLD (expected daily demand).
    forecast_model : the forecast the agent OBSERVES (default: same model).
    Same model for both = "ML1 + PPO" or "ML2 + PPO", each in its own world.
    A common world with different forecasts isolates which model helps
    pricing more.

PRICES IN THIS VERSION
    "Price" = average recorded price per room-night (as in the export, no
    BAR reconstruction). The agent chooses it as a multiple of the season's
    reference (the hotel's median: ~EUR 129 in 2025, ~EUR 133 in 2026).
    A later phase will switch to the reconstructed base rate (BAR).

WHAT HAPPENS IN ONE STEP (arrival date t, price p chosen by the agent)
  1. Expected booking requests
         lambda = lambda_base(t) * g(p / p_ref)
     lambda_base(t): Random Forest demand at the hotel's own price, with the
     (small) effect of the hotel's actual price on that date removed.
  2. Number of requests ~ Negative Binomial(lambda, k): Poisson plus the
     extra day-to-day variability measured in the data.
  3. Each request is a REAL booking of the same season and month,
     resampled from the hotel's data, keeping its length of stay, rooms,
     real price paid (Total) and whether it was cancelled.
  4. Cancelled requests bring no revenue and take no room. Others take a
     room for every night of their stay if one is free; otherwise lost.
  5. Revenue = the booking's real Total, scaled by how much the agent's
     price differs from the hotel's price on that booking's own date:
         revenue = Total x p / orig_price
     So at the hotel's prices the simulator pays exactly what guests paid,
     and discounts / channels / occupancy surcharges are kept implicitly.

  COMMON RANDOM NUMBERS: all randomness of a season (daily demand shocks,
  which guests appear, what they book, whether they cancel) is drawn at
  reset(), independently of price. Every policy is tested on the SAME season
  with the SAME guests; only the prices differ — the exact "what if the
  hotel had priced differently?" question, with little comparison noise.

REWARD = revenue of the bookings accepted that day, minus an optional
variable cost per room-night (default 0: costs are not known).

PRICE RESPONSE (ELASTICITY) — the one assumption not taken from the data
    The hotel priced almost flat, so its data cannot reveal how demand
    reacts to price. Scenario parameter:
        g(r) = exp(-ELASTICITY * (r - 1)),   r = price / season reference
    Default 1.0: the hotel's own average price level is revenue-optimal when
    rooms are not scarce — deliberately conservative, so any gain comes from
    adapting the price day by day, not from assuming the hotel's level was
    wrong. Literature: elasticities of online resort-hotel demand differ
    strongly by hotel and season (Vives, Jacob & Aguiló, 2019).

OBSERVATION (before pricing date t), all in [-1, 1]:
    demand forecast (out-of-fold RF), rooms already taken on t and t+1,
    month (sin/cos), weekday (sin/cos), holiday flag, season progress.

SIMPLIFICATIONS (stated, not hidden)
    - A stay is priced at the level of its arrival date for all nights.
    - Capacity = all 17 STD rooms for online demand (offline / B2B bookings
      also use rooms in reality, so true scarcity is higher).
    - Choice of rate plan and channel does not depend on price.

Author    : IntelliRate — Σπηλιόπουλος Σπήλιος  AM 19153
Supervisor: Αθανάσιος Κούτρας, University of Peloponnese 2026
"""

import numpy as np
import joblib
import gymnasium as gym
from gymnasium import spaces


SIM_DATA_PATH = "sim_data.pkl"

ELASTICITY    = 1.0          # price elasticity at the reference price (scenario)
VARIABLE_COST = 0.0          # EUR per sold room-night (unknown -> 0 = revenue)

# Price levels as multiples of the season reference (+/-25%)
PRICE_MULTIPLIERS = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
                     1.05, 1.10, 1.15, 1.20, 1.25]

MAX_REQUESTS = 80            # per day; far above any realistic demand here


def poisson_from_uniform(u, lam):
    """Inverse-CDF Poisson draw: the same u gives more requests when lam is
    higher, so a lower price can never produce fewer guests (monotone)."""
    k, p = 0, np.exp(-lam)
    cdf = p
    while u > cdf and k < MAX_REQUESTS:
        k += 1
        p *= lam / k
        cdf += p
    return k


class HotelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, season=2025, demand_model="rf", forecast_model=None,
                 elasticity=ELASTICITY, variable_cost=VARIABLE_COST,
                 capacity=None, sim_path=SIM_DATA_PATH):
        super().__init__()
        data = joblib.load(sim_path)
        s = data["seasons"][season]
        forecast_model = forecast_model or demand_model
        self.demand_model, self.forecast_model = demand_model, forecast_model

        self.season        = season
        self.elasticity    = elasticity
        self.variable_cost = variable_cost
        self.capacity      = capacity or data["capacity"]
        self.k             = data["dispersion_k"][demand_model]   # None -> Poisson

        self.dates       = s["dates"]
        self.n_days      = len(self.dates)
        self.p_ref       = s["p_ref"]
        self.hotel_price = s["hotel_price"]
        self.forecast    = s["lambda_forecast"][forecast_model]
        self.month       = s["month"]
        self.dow         = s["dow"]
        self.holiday     = s["holiday"]
        self.actual      = s["actual"]
        self.prices      = self.p_ref * np.array(PRICE_MULTIPLIERS)

        # Demand at the reference price (hotel's own price effect removed)
        self.lambda_base = s["lambda_true"][demand_model] / self._g(self.hotel_price)
        self.forecast_scale = float(np.max(self.forecast))

        # Booking profiles as plain arrays (fast sampling)
        self.pools = {
            m: dict(rooms=df["rooms"].to_numpy(int),
                    nights=df["nights"].to_numpy(int),
                    total=df["Total"].to_numpy(float),
                    orig_price=df["orig_price"].to_numpy(float),
                    cancelled=df["is_cancelled"].to_numpy(bool))
            for m, df in s["profiles"].items()
        }

        self.action_space = spaces.Discrete(len(PRICE_MULTIPLIERS))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(9,), dtype=np.float32)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _g(self, price):
        return np.exp(-self.elasticity * (np.asarray(price) / self.p_ref - 1.0))

    def _obs(self):
        t = min(self.t, self.n_days - 1)
        m, d = self.month[t], self.dow[t]
        return np.array([
            2.0 * self.forecast[t] / self.forecast_scale - 1.0,
            2.0 * self.committed[t] / self.capacity - 1.0,
            2.0 * self.committed[t + 1] / self.capacity - 1.0,
            np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12),
            np.sin(2 * np.pi * d / 7),  np.cos(2 * np.pi * d / 7),
            2.0 * self.holiday[t] - 1.0,
            2.0 * t / self.n_days - 1.0,
        ], dtype=np.float32).clip(-1.0, 1.0)

    # ── gymnasium API ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.t = 0
        self.committed = np.zeros(self.n_days + 40, dtype=int)
        # The whole season's randomness, fixed before any price is set
        self.shock = (rng.gamma(self.k, 1.0 / self.k, size=self.n_days)
                      if self.k else np.ones(self.n_days))
        self.u_count = rng.random(self.n_days)
        self.u_pick = rng.random((self.n_days, MAX_REQUESTS))
        return self._obs(), {}

    def step(self, action):
        return self.step_price(self.prices[int(action)])

    def step_price(self, price):
        """Simulate arrival date t at an explicit price (used by baselines)."""
        t = self.t
        lam = self.lambda_base[t] * self._g(price) * self.shock[t]
        n_requests = poisson_from_uniform(self.u_count[t], lam)

        pool = self.pools[int(self.month[t])]
        picks = (self.u_pick[t, :n_requests] * len(pool["rooms"])).astype(int)

        revenue, room_nights, accepted, cancelled, lost = 0.0, 0, 0, 0, 0
        for i in picks:
            if pool["cancelled"][i]:
                cancelled += 1
                continue
            rooms, nights = pool["rooms"][i], pool["nights"][i]
            if self.committed[t:t + nights].max() + rooms > self.capacity:
                lost += 1
                continue
            self.committed[t:t + nights] += rooms
            revenue += pool["total"][i] * price / pool["orig_price"][i]
            room_nights += rooms * nights
            accepted += 1

        reward = revenue - self.variable_cost * room_nights
        info = dict(date=self.dates[t], price=float(price), requests=int(n_requests),
                    accepted=accepted, cancelled=cancelled, lost=lost,
                    revenue=revenue, room_nights=room_nights,
                    occupied=int(self.committed[t]))
        self.t += 1
        terminated = self.t >= self.n_days
        return self._obs(), float(reward), terminated, False, info
