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
