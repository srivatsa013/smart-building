import numpy as np
import random


class BuildingEnv:
    """
    Smart Building Environment for Reinforcement Learning.

    State : (temp_bin, occupancy, time_bin, price_bin)
    Action: 0 = AC OFF, 1 = AC ON
    Reward: -(cost + COMFORT_WEIGHT * comfort_penalty)

    Training mode  (training=True)       -> randomised start + noisy conditions
    Simulation mode (initial_temp=value) -> fixed, deterministic for the app

    Temperature binning covers 20–45°C (Vellore summer range) in 20 bins of 1.25°C each.
    """

    # Realistic Vellore summer day profile (April–May):
    # Min ~25°C at 3–4 am, peak ~41°C at noon–1 pm.
    OUTDOOR_TEMPS = [
        27, 26, 26, 25, 25, 26, 28, 31,
        34, 37, 39, 40, 41, 41, 40, 39,
        37, 35, 33, 31, 30, 29, 28, 27,
    ]

    PRICE_PROFILE = [
        2, 2, 2, 2, 2, 3, 4, 6,
        7, 8, 8, 9, 9, 8, 8, 7,
        7, 8, 9, 8, 6, 4, 3, 2,
    ]

    # Raise this -> agent cares more about comfort vs saving money
    COMFORT_WEIGHT = 2.5

    # Probability per step that occupancy flips (training only)
    OCC_FLIP_PROB = 0.08

    def __init__(self, initial_temp=None, initial_occupancy=None, training=False,
                 outdoor_temps=None, price_profile=None):
        self.initial_temp      = initial_temp
        self.initial_occupancy = initial_occupancy
        self.training          = training
        # Support custom profiles for dynamic/API mode
        self._outdoor_temps = outdoor_temps if outdoor_temps is not None else self.OUTDOOR_TEMPS
        self._price_profile = price_profile if price_profile is not None else self.PRICE_PROFILE
        self.reset()

    def reset(self):
        if self.training:
            # Cover the FULL state space so every Q-table cell gets visited
            self.indoor_temp     = random.uniform(20, 43)
            self.occupancy_level = random.random()
            self.occupancy       = random.randint(0, 1)
            self.time            = random.randint(0, 23)
        else:
            self.indoor_temp = self.initial_temp if self.initial_temp is not None else random.uniform(26, 34)
            occ = self.initial_occupancy if self.initial_occupancy is not None else 1.0
            self.occupancy_level = float(occ)
            self.occupancy       = 1 if self.occupancy_level >= 0.5 else 0
            self.time            = 0   # Always start at hour 0 for a full 24-hour simulation

        self.price = self._price_profile[self.time % 24]
        return self.get_state()

    def get_state(self):
        # temp_bin covers 20–45°C in 20 bins of 1.25°C each
        temp_bin  = int(np.clip((self.indoor_temp - 20) / 1.25, 0, 19))
        time_bin  = min(self.time // 3, 7)
        price_bin = int(np.clip((self.price - 2) // 1.5, 0, 5))
        return (temp_bin, self.occupancy, time_bin, price_bin)

    def step(self, action):
        hour        = self.time % 24
        outdoor_raw = self._outdoor_temps[hour]
        price_raw   = self._price_profile[hour]

        # Add noise during training -> agent learns a GENERAL policy
        if self.training:
            outdoor_temp = outdoor_raw + random.uniform(-2.0, 2.0)
            self.price   = max(1, price_raw + random.uniform(-1.0, 1.0))
        else:
            outdoor_temp = outdoor_raw
            self.price   = price_raw

        # Energy model
        if action == 1:
            self.indoor_temp -= 1.2
            energy = 1.5
        else:
            self.indoor_temp += 0.25 * (outdoor_temp - self.indoor_temp)
            energy = 0.0

        # Clamp to realistic indoor range for Vellore conditions
        self.indoor_temp = float(np.clip(self.indoor_temp, 15, 45))

        # Comfort penalty (target 22–24°C)
        # occ_multiplier scales linearly: 0% occupancy → 0.5, 100% → 2.0
        raw_comfort     = max(0.0, self.indoor_temp - 24) + max(0.0, 22 - self.indoor_temp)
        occ_multiplier  = 0.5 + self.occupancy_level * 1.5
        comfort_penalty = raw_comfort * occ_multiplier

        # Cost
        cost = energy * self.price

        # Reward - COMFORT_WEIGHT balances cost vs comfort
        reward = -(cost + self.COMFORT_WEIGHT * comfort_penalty)

        self.time += 1
        done = self.time >= 24

        # Stochastic occupancy flip during training
        if self.training and random.random() < self.OCC_FLIP_PROB:
            self.occupancy       = 1 - self.occupancy
            self.occupancy_level = float(self.occupancy)

        return self.get_state(), reward, done, energy, self.indoor_temp, cost, comfort_penalty


def rule_based_action(temp):
    """Simple threshold controller: cool if above 25°C."""
    return 1 if temp > 25 else 0
