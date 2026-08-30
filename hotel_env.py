"""
IntelliRate — Hotel Pricing Environment (ML + DL Version)
============================================================
Phase 2: The environment now uses a REAL demand predictor —
a Logistic Regression trained on the Kaggle Hotel Booking Demand
dataset — instead of the hand-crafted sigmoid formula.

ARCHITECTURE:
    Kaggle CSV
        → Logistic Regression (scikit-learn)   [ML — trained offline]
        → demand_model.pkl                     [saved predictor]
        → loaded HERE, inside the environment
        → PPO / DQN (Stable-Baselines3)         [DL — trained here]

WHAT CHANGED FROM THE DUMMY VERSION:
    - _booking_probability() now calls the trained Logistic Regression
      instead of a sigmoid formula.
    - SEASONALITY and DAY_MULTIPLIERS are REMOVED as separate config —
      they are no longer needed because the Logistic Regression already
      learned seasonal and weekend patterns directly from real Kaggle
      data (via the month dummy variables and is_weekend feature used
      during training). Re-applying them here would double-count the
      effect.
    - OCCUPANCY_BOOST_WEIGHT is KEPT — occupancy is hotel-specific,
      real-time state that the Kaggle dataset (booking-level records)
      cannot capture, so it remains a separate environment-level signal.

WHAT STAYED THE SAME:
    - Action space (10 price multipliers)
    - Reward function (RevPAR)
    - State space structure
    - PPO / DQN training code — completely unchanged

Author    : IntelliRate — Σπηλιόπουλος Σπήλιος  AM 19153
Supervisor: Αθανάσιος Κούτρας, University of Peloponnese 2026
"""

import numpy as np
import pandas as pd
import joblib
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# PARAMETERS
# =============================================================================

TOTAL_ROOMS    = 17          # Standard/Classic rooms in the real hotel
BASE_ADR       = 131.0       # mean ADR (€) for Standard room — used as
                              # the CENTRE of the price multiplier range,
                              # NOT fed into the demand model as a default;
                              # actual prices always come from the action.
EPISODE_LENGTH = 90          # days per training episode

# 10 discrete price multipliers
PRICE_MULTIPLIERS = [0.70, 0.80, 0.85, 0.90, 0.95,
                     1.00, 1.05, 1.10, 1.25, 1.50]

# Occupancy still boosts demand — this is hotel-specific real-time state,
# not something the Kaggle demand model can know about.
OCCUPANCY_BOOST_WEIGHT = 0.25

# Paths to the files produced by train_demand_model.py
DEMAND_MODEL_PATH    = "demand_model.pkl"
DEMAND_FEATURES_PATH = "demand_features.pkl"


# =============================================================================
# ENVIRONMENT
# =============================================================================

class HotelEnv(gym.Env):
    """
    Hotel Room Dynamic Pricing Environment — ML + DL version.

    STATE (4 values, all in [-1, 1]):
        [occupancy_rate, month_sin, month_cos, is_weekend]

    ACTION:
        Integer 0–9 → one of 10 PRICE_MULTIPLIERS
        Actual price = multiplier × BASE_ADR

    REWARD:
        RevPAR = (actual_price × rooms_booked) / TOTAL_ROOMS

    DEMAND MODEL:
        P(booking | price, lead_time, month, is_weekend) predicted by
        a Logistic Regression trained on real Kaggle hotel data.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None,
                model_path=DEMAND_MODEL_PATH,
                features_path=DEMAND_FEATURES_PATH):
        super().__init__()
        self.render_mode = render_mode

        # ── LOAD THE TRAINED ML PREDICTOR ─────────────────────────────────
        # This is the connection point between the ML stage and the RL
        # environment. Loaded ONCE here, then called every step().
        self.demand_model     = joblib.load(model_path)
        self.feature_columns  = joblib.load(features_path)

        # ── Action space ──────────────────────────────────────────────────
        self.action_space = spaces.Discrete(len(PRICE_MULTIPLIERS))

        # ── Observation space ─────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        # Internal state
        self.current_day   = 0
        self.current_month = 6
        self.lead_time      = 30     # used as a feature for the ML model
        self.occupancy      = 0.0
        self.total_revenue  = 0.0

    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_day   = 0
        self.total_revenue = 0.0
        self.current_month = int(self.np_random.integers(5, 8))  # Jun–Aug
        self.occupancy = float(np.clip(
            self.np_random.normal(0.20, 0.10), 0.0, 1.0
        ))

        # Lead time proxy: how far ahead the "typical" booking decision is
        # made relative to today. Randomised per episode so the agent sees
        # a range of booking horizons, same spirit as the dummy version.
        self.lead_time = int(self.np_random.integers(0, 90))

        return self._observe(), {}

    # ─────────────────────────────────────────────────────────────────────
    def step(self, action):

        multiplier   = PRICE_MULTIPLIERS[action]
        actual_price = BASE_ADR * multiplier

        # ── Booking probability from the trained ML model ────────────────
        prob = self._booking_probability(actual_price)

        booked       = float(self.np_random.random() < prob)
        available    = max(0.0, TOTAL_ROOMS * (1.0 - self.occupancy))
        rooms_booked = min(available,
                          booked * float(self.np_random.integers(1, TOTAL_ROOMS + 1)))

        daily_revenue = actual_price * rooms_booked
        reward        = daily_revenue / TOTAL_ROOMS

        checkout_rate  = 0.60      # avg LOS < 2 days → checkout_rate ≈ 0.60
        self.occupancy = float(np.clip(
            self.occupancy
            + (rooms_booked / TOTAL_ROOMS)
            - (self.occupancy * checkout_rate),
            0.0, 1.0
        ))

        self.current_day   += 1
        self.total_revenue += daily_revenue
        self.lead_time       = max(0, self.lead_time - 1)  # counts down

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
            "booking_prob" : round(prob, 4),
            "month"        : self.current_month + 1,
        }

        return self._observe(), reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────
    def _booking_probability(self, actual_price):
        """
        P(booking | price, state) — predicted by the trained
        Logistic Regression, NOT a hand-crafted formula.

        This is the ML → DL connection point:
            1. Build a feature row matching the columns the model
               was trained on (adr, lead_time, is_weekend, 12 month
               dummies).
            2. Call model.predict_proba() to get the base probability
               the market would book at this price, given these
               conditions.
            3. Add an occupancy boost — hotel-specific real-time
               state the Kaggle model cannot know about.
        """

        # Build a single-row feature vector, all zeros by default,
        # matching the exact columns used during training.
        row = {col: 0 for col in self.feature_columns}
        row["adr"]        = actual_price
        row["lead_time"]  = self.lead_time
        row["is_weekend"] = float((self.current_day % 7) >= 5)

        month_col = f"month_{self.current_month + 1}"   # 1-indexed to match training
        if month_col in row:
            row[month_col] = 1

        X = pd.DataFrame([row])[self.feature_columns]

        # predict_proba returns [[P(class=0), P(class=1)]] — we want
        # P(class=1) = P(booked)
        base_prob = float(self.demand_model.predict_proba(X)[0][1])

        # Occupancy boost — real-time hotel state, added on top of the
        # market-level prediction from the ML model.
        occ_boost = OCCUPANCY_BOOST_WEIGHT * self.occupancy

        return float(np.clip(base_prob + occ_boost, 0.05, 0.95))

    # ─────────────────────────────────────────────────────────────────────
    def _observe(self):
        obs_occ = (self.occupancy * 2.0) - 1.0

        angle         = 2.0 * np.pi * self.current_month / 12.0
        obs_month_sin = float(np.sin(angle))
        obs_month_cos = float(np.cos(angle))

        is_weekend  = float((self.current_day % 7) >= 5)
        obs_weekend = (is_weekend * 2.0) - 1.0

        return np.array(
            [obs_occ, obs_month_sin, obs_month_cos, obs_weekend],
            dtype=np.float32,
        )

    # ─────────────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode == "human":
            print(
                f"Day {self.current_day:3d} | "
                f"Month {self.current_month+1:2d} | "
                f"Occupancy {self.occupancy*100:5.1f}% | "
                f"Revenue so far €{self.total_revenue:.2f}"
            )
