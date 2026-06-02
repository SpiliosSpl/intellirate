import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Hotel calibrated to Standard/Classic room - from the hotel data
TOTAL_ROOMS    = 17
BASE_ADR       = 131.0 # mean Average Daily Rate for STD room type
EPISODE_LENGTH = 90 # days per training episode (3 months)

# ===========PRICE MULTIPLIERS==================
# Agent picks one of the 10 levels every day
# Actual price = multiplier x BASE_ADR
PRICE_MULTIPLIERS = [0.70, 0.80, 0.85, 0.90, 0.95, 
                     1.00, 1.05, 1.10, 1.25, 1.50]

# ===========SEASONALITY MULTIPLIERS==============
# Jan, Feb, Mar, Apr, May, Jun, Jul, Aug (1.00 -> peak season), Sep, Oct, Nov, Dec
SEASONALITY = [0.00, 0.00, 0.05, 0.20, 0.45, 0.75,
               0.95, 1.00, 0.85, 0.50, 0.05, 0.00]

# ==========DAY TYPE MULTIPLIERS=================
# weekends have higher demand
DAY_MULTIPLIERS = {
    "weekday" : 1.00,
    "weekend" : 1.15,
}

# ==========OCCUPANCY BOOST======================
# High occupancy boosts demand signal - dummy simulation of FOMO effect
OCCUPANCY_BOOST_WEIGHT = 0.25

class HotelEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        #======== Action space =========
        # 10 discrete price actions
        self.action_space = spaces.Discrete(len(PRICE_MULTIPLIERS))

        #==========Observation space ==============
        # 4 values-> occupancy, month (sin+cos), is_weekend
        self.observation_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (4,),
            dtype = np.float32,
        )

        # internal state
        self.current_day   = 0
        self.current_month = 6
        self.occupancy     = 0.0
        self.total_revenue = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_day   = 0
        self.total_revenue = 0.0

        self.current_month = int(self.np_random.integers(5, 8))

        # Randomise starting occupancy (10–35%)
        self.occupancy = float(np.clip(
            self.np_random.normal(0.20, 0.10), 0.0, 1.0
        ))

        return self._observe(), {}

    def step(self, action):
        action = int(action)

        #actual price
        multiplier   = PRICE_MULTIPLIERS[action]
        actual_price = BASE_ADR * multiplier

        #Booking probability
        prob = self._booking_probability(actual_price)

        #Simulate booking
        booked = float(self.np_random.random() < prob)
        available = max(0.0, TOTAL_ROOMS * (1.0 - self.occupancy))
        rooms_booked = min(available,
                           booked * float(self.np_random.integers(1, TOTAL_ROOMS + 1)))

        #RevPAR reward
        daily_revenue = actual_price * rooms_booked
        reward = daily_revenue / TOTAL_ROOMS

        # Update occupancy
        # Checkout rate = 1 / average_LOS
        checkout_rate = 0.70 # 1 / 0.70 = 1.43 days average LOS
        self.occupancy = float(np.clip(
            self.occupancy
            + (rooms_booked / TOTAL_ROOMS)
            - (self.occupancy * checkout_rate),
            0.0, 1.0
        ))

        self.current_day   += 1
        self.total_revenue += daily_revenue

        # Advance month every 30 days
        if self.current_day % 30 == 0:
            self.current_month = (self.current_month + 1) % 12

        terminated = self.current_day >= EPISODE_LENGTH
        truncated  = False

        info = {
            "multiplier"   : multiplier,
            "actual_price" : round(actual_price, 2),
            "rooms_booked" : rooms_booked,
            "occupancy"    : round(self.occupancy, 3),
            "reward"       : round(reward, 3),
            "month"        : self.current_month + 1,
        }

        return self._observe(), reward, terminated, truncated, info


    def _booking_probability(self, actual_price):
        # Price sensitivity - sigmoid
        steepness = BASE_ADR / 3.0 #3.0 so it auto-scales with regardless of BASE_ADR
        price_effect = 1.0 / (1.0 + np.exp((actual_price - BASE_ADR) / steepness))

        # Seasonality
        season = SEASONALITY[self.current_month]

        # Day type: weekend / weekday
        is_weekend = (self.current_day % 7) >= 5
        day_mult   = DAY_MULTIPLIERS["weekend"] if is_weekend \
                     else DAY_MULTIPLIERS["weekday"]

        # Occupancy boost
        occ_boost = OCCUPANCY_BOOST_WEIGHT * self.occupancy

        # Combine price and occupancy scaled by season and day type
        prob = (price_effect + occ_boost) * season * day_mult

        return float(np.clip(prob, 0.05, 0.95))


    def _observe(self):

        obs_occ = (self.occupancy * 2.0) - 1.0 #convert occupancy from [0,1] to [-1,1]

        # Cyclical months so December and January are next to each other
        angle = 2.0 * np.pi * self.current_month / 12.0
        obs_month_sin = float(np.sin(angle))
        obs_month_cos = float(np.cos(angle))

        # Weekend flag: 1=weekend, 0=weekday → [-1,1]
        is_weekend  = float((self.current_day % 7) >= 5)
        obs_weekend = (is_weekend * 2.0) - 1.0

        return np.array(
            [obs_occ, obs_month_sin, obs_month_cos, obs_weekend],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode == "human":
            print(
                f"Day {self.current_day:3d} | "
                f"Month {self.current_month+1:2d} | "
                f"Occupancy {self.occupancy*100:5.1f}% | "
                f"Revenue so far €{self.total_revenue:.2f}"
            )
