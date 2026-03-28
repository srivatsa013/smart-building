"""
train.py - Q-Learning trainer for Smart Building RL
Run: python train.py
Saves: q_table.npy, rewards_log.npy
"""

import numpy as np
import random
from env import BuildingEnv

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPISODES      = 50_000      # 3x more -> cold/uncommon states get visited
ALPHA         = 0.1         # learning rate
GAMMA         = 0.95        # discount factor
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9999      # slower decay: hits 0.05 at ~ep 19,000
                            # -> last 11k episodes are quality exploitation

# Q-table: (temp_bin=20, occupancy=2, time_bin=8, price_bin=6, actions=2)
# Total cells: 20 x 2 x 8 x 6 x 2 = 3,840
# With 50k eps x 24 steps = 1,200,000 updates -> ~300 updates per cell on average
Q = np.zeros((20, 2, 8, 6, 2))

epsilon    = EPSILON_START
env        = BuildingEnv(training=True)   # <-- training=True enables randomisation
rewards_log = []

print(f"Training for {EPISODES:,} episodes...")
print(f"  State space: 20 x 2 x 8 x 6 x 2 = {20*2*8*6*2:,} cells")
print(f"  Total steps: {EPISODES * 24:,}  (~{EPISODES * 24 // (20*2*8*6*2)} updates/cell avg)\n")

for ep in range(EPISODES):
    state    = env.reset()
    done     = False
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

        nt_bin, nocc, nti_bin, np_bin = next_state   # np_bin != numpy

        # Q-Learning (Bellman) update
        old_val  = Q[t_bin,  occ,  ti_bin,  p_bin,  action]
        next_max = np.max(Q[nt_bin, nocc, nti_bin, np_bin])

        Q[t_bin, occ, ti_bin, p_bin, action] = old_val + ALPHA * (
            reward + GAMMA * next_max - old_val
        )

        state = next_state

    # Slow epsilon decay
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    rewards_log.append(ep_reward)

    if (ep + 1) % 5000 == 0:
        avg    = np.mean(rewards_log[-5000:])
        cells_nonzero = int(np.sum(Q != 0) / 2)   # /2 because 2 actions
        print(f"  Episode {ep+1:>6,} | Avg reward: {avg:>7.2f} | "
              f"epsilon: {epsilon:.4f} | "
              f"Q-cells learned: {cells_nonzero:,}/{20*2*8*6}")

np.save("q_table.npy", Q)
np.save("rewards_log.npy", np.array(rewards_log))

# Final quality check
final_avg = np.mean(rewards_log[-5000:])
early_avg = np.mean(rewards_log[:5000])
print(f"\n{'='*55}")
print(f"  Training complete!")
print(f"  Early avg reward  (ep 1-5k)   : {early_avg:.2f}")
print(f"  Final avg reward  (ep 25-30k) : {final_avg:.2f}")
print(f"  Improvement                   : {final_avg - early_avg:+.2f}")
print(f"  Q-cells with learned values   : {int(np.sum(Q != 0)/2):,} / {20*2*8*6}")
print(f"{'='*55}")
print(f"\nSaved: q_table.npy, rewards_log.npy")