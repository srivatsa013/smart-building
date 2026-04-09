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
    """

    OUTDOOR_TEMPS = [
        22, 21, 21, 20, 20, 21, 23, 26,
        28, 30, 32, 33, 34, 34, 33, 32,
        31, 30, 28, 27, 26, 25, 24, 23,
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

    def __init__(self, initial_temp=None, initial_occupancy=None, training=False):
        self.initial_temp      = initial_temp
        self.initial_occupancy = initial_occupancy
        self.training          = training
        self.reset()

    def reset(self):
        if self.training:
            # Cover the FULL state space so every Q-table cell gets visited
            self.indoor_temp = random.uniform(18, 36)
            self.occupancy   = random.randint(0, 1)
        else:
            self.indoor_temp = self.initial_temp if self.initial_temp is not None else random.uniform(24, 30)
            self.occupancy   = self.initial_occupancy if self.initial_occupancy is not None else random.randint(0, 1)

        self.time  = random.randint(0, 23)
        self.price = self.PRICE_PROFILE[0]
        return self.get_state()

    def get_state(self):
        temp_bin  = int(np.clip(self.indoor_temp - 18, 0, 19))
        time_bin  = min(self.time // 3, 7)
        price_bin = int(np.clip((self.price - 2) // 1.5, 0, 5))
        return (temp_bin, self.occupancy, time_bin, price_bin)

    def step(self, action):
        hour        = self.time % 24
        outdoor_raw = self.OUTDOOR_TEMPS[hour]
        price_raw   = self.PRICE_PROFILE[hour]

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

        self.indoor_temp = float(np.clip(self.indoor_temp, 15, 40))

        # Comfort penalty (target 22-24 degC)
        raw_comfort     = max(0.0, self.indoor_temp - 24) + max(0.0, 22 - self.indoor_temp)
        occ_multiplier  = 2.0 if self.occupancy == 1 else 0.5
        comfort_penalty = raw_comfort * occ_multiplier

        # Cost
        cost = energy * self.price

        # Reward - COMFORT_WEIGHT balances cost vs comfort
        reward = -(cost + self.COMFORT_WEIGHT * comfort_penalty)

        self.time += 1
        done = self.time >= 24

        # Stochastic occupancy flip during training
        if self.training and self.initial_occupancy is None:
            if random.random() < self.OCC_FLIP_PROB:
                self.occupancy = 1 - self.occupancy

        return self.get_state(), reward, done, energy, self.indoor_temp, cost, comfort_penalty


def rule_based_action(temp):
    """Simple threshold controller: cool if above 25 degC."""
    return 1 if temp > 25 else 0