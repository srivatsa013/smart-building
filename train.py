"""
train.py - Q-Learning trainer for Smart Building RL
Run: python train.py
Saves: q_table.npy, rewards_log.npy

For real weather data training, first run:
    python fetch_weather_profiles.py
Then re-run this script. If weather_profiles.npy is present it will be used
automatically; otherwise falls back to the built-in synthetic profile.
"""

import os
import numpy as np
import random
from env import BuildingEnv

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPISODES      = 300_000
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.99995

# ── Load real weather profiles if available ──────────────────────────────────
_PROFILES_FILE = "weather_profiles.npy"
if os.path.exists(_PROFILES_FILE):
    weather_profiles = np.load(_PROFILES_FILE)   # shape (N_days, 24)
    print(f"Real weather data loaded: {weather_profiles.shape[0]} days from {_PROFILES_FILE}")
    USE_REAL_WEATHER = True
else:
    weather_profiles = None
    USE_REAL_WEATHER = False
    print("No weather_profiles.npy found — using built-in synthetic profile.")
    print("Run `python fetch_weather_profiles.py` first for real Vellore data.\n")

# Q-table: (temp_bin=20, occupancy=2, time_bin=8, price_bin=6, actions=2)
# Total cells: 20 x 2 x 8 x 6 x 2 = 3,840
Q = np.zeros((20, 2, 8, 6, 2))

epsilon     = EPSILON_START
rewards_log = []

print(f"Training for {EPISODES:,} episodes...")
print(f"  State space : 20 × 2 × 8 × 6 = {20*2*8*6:,} states")
print(f"  Total steps : {EPISODES * 24:,}  (~{EPISODES * 24 // (20*2*8*6)} updates/cell avg)")
print(f"  Weather     : {'Real (Vellore historical)' if USE_REAL_WEATHER else 'Synthetic'}\n")

for ep in range(EPISODES):
    # Pick an outdoor temperature profile for this episode
    if USE_REAL_WEATHER:
        day_profile = weather_profiles[ep % len(weather_profiles)].tolist()
        env = BuildingEnv(training=True, outdoor_temps=day_profile)
    else:
        env = BuildingEnv(training=True)

    state     = env.reset()
    done      = False
    ep_reward = 0.0

    while not done:
        t_bin, occ, ti_bin, p_bin = state

        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = int(np.argmax(Q[t_bin, occ, ti_bin, p_bin]))

        next_state, reward, done, *_ = env.step(action)
        ep_reward += reward

        nt_bin, nocc, nti_bin, np_bin = next_state

        # Q-Learning (Bellman) update
        old_val  = Q[t_bin,  occ,  ti_bin,  p_bin,  action]
        next_max = np.max(Q[nt_bin, nocc, nti_bin, np_bin])
        Q[t_bin, occ, ti_bin, p_bin, action] = old_val + ALPHA * (
            reward + GAMMA * next_max - old_val
        )

        state = next_state

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    rewards_log.append(ep_reward)

    if (ep + 1) % 5000 == 0:
        avg           = np.mean(rewards_log[-5000:])
        cells_nonzero = int(np.sum(Q != 0) / 2)
        print(f"  Episode {ep+1:>6,} | Avg reward: {avg:>7.2f} | "
              f"epsilon: {epsilon:.4f} | Q-cells learned: {cells_nonzero:,}/{20*2*8*6}")

np.save("q_table.npy",    Q)
np.save("rewards_log.npy", np.array(rewards_log))

final_avg = np.mean(rewards_log[-5000:])
early_avg = np.mean(rewards_log[:5000])
print(f"\n{'='*55}")
print(f"  Training complete!")
print(f"  Early avg reward  (ep 1–5k)    : {early_avg:.2f}")
print(f"  Final avg reward  (ep 95–100k) : {final_avg:.2f}")
print(f"  Improvement                    : {final_avg - early_avg:+.2f}")
print(f"  Q-cells with learned values    : {int(np.sum(Q != 0)/2):,} / {20*2*8*6}")
print(f"{'='*55}")
print(f"\nSaved: q_table.npy, rewards_log.npy")
