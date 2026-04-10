# Smart Building RL — Intelligent HVAC Control

A Q-Learning agent that learns to control HVAC in a smart building, balancing **energy cost** and **occupant comfort** across a 24-hour cycle. The dashboard compares the trained RL agent side-by-side against a simple rule-based baseline.

---

## Motivation

Buildings account for roughly 40% of global energy consumption, with HVAC systems responsible for the bulk of that. Traditional rule-based controllers — *"turn AC on when temperature exceeds 25°C"* — are simple but wasteful: they react to the current temperature without considering electricity pricing, whether the space is occupied, or what the temperature will be in a few hours.

Reinforcement Learning offers a smarter alternative. An agent trained over hundreds of thousands of simulated days learns to:

- **Pre-cool rooms before peak-price hours** (6–9 pm in India), so less cooling is needed when electricity is most expensive.
- **Delay or skip cooling when the building is empty**, since human comfort only matters when people are present.
- **Let natural heat exchange do the work at night**, when outdoor air is cool enough to passively bring the room into the comfort band without running the AC at all.

---

## How It Works

### Environment

The simulation models a single-zone building with the following state and dynamics:

| Component | Details |
|-----------|---------|
| **State** | Indoor temp (18–38 °C → 20 bins), occupancy (0/1), time of day (8 bins), electricity price (6 bins) = **3,840 unique states** |
| **Actions** | AC OFF (0) or AC ON (1) |
| **AC physics** | ON → indoor temp drops 1.2 °C/hour; OFF → temp shifts toward outdoor via Newton's law of cooling |
| **Comfort band** | 22–24 °C; penalty grows linearly outside this band, doubled when occupied |
| **Electricity pricing** | India residential time-of-use tariff — off-peak night (₹3/unit), daytime (₹5–7), evening peak 6–9 pm (₹9) |

**Reward signal** (what the agent minimises):

```
reward = −( energy_cost  +  2.5 × comfort_penalty )
```

A higher `COMFORT_WEIGHT` makes the agent sacrifice more cost savings to keep occupants comfortable. The factor of 2× on occupied hours means the agent learns to care far more about comfort when people are present.

---

### Q-Learning Algorithm

Q-Learning is a **model-free, off-policy** reinforcement learning algorithm. It maintains a table of Q-values — one per (state, action) pair — representing the expected cumulative reward of taking that action from that state.

After every environment step, the table is updated via the **Bellman equation**:

```
Q(s, a)  ←  Q(s, a)  +  α [ r  +  γ · max_a' Q(s', a')  −  Q(s, a) ]
```

| Symbol | Value | Meaning |
|--------|-------|---------|
| α | 0.1 | Learning rate — how much each new experience overwrites old estimates |
| γ | 0.95 | Discount factor — future rewards count at 95% of immediate ones |
| ε (start→end) | 1.0 → 0.05 | Epsilon-greedy exploration: starts fully random, decays to 5% over ~60k episodes |

**Training:** 100,000 episodes × 24 hours = **2.4 million environment steps**, giving roughly 600 updates per Q-table cell on average. Training is run once (`python train.py`) and the resulting `q_table.npy` is loaded by the dashboard at runtime.

---

### Rule-Based Baseline

```python
def rule_based_action(temp):
    return 1 if temp > 25 else 0   # AC ON above 25 °C, OFF otherwise
```

This is the naive industry default. It reacts only to current temperature and ignores price, occupancy, or time of day — making it the natural benchmark to beat.

---

## Results

After training, the RL agent consistently outperforms the rule-based controller on at least one axis across most input scenarios:

- **Cost savings** — RL learns to avoid running AC during the expensive 6–9 pm peak window, shifting cooling to cheaper hours even if that means a slightly warmer room at peak time.
- **Comfort improvement** — RL anticipates rising afternoon temperatures and pre-cools earlier, whereas the rule-based controller only reacts once the room is already too warm.
- **Fewer AC hours** — By exploiting natural overnight cooling (outdoor air 20–22 °C), RL often achieves the same or better comfort with fewer hours of active cooling.

The dashboard shows the full 24-hour comparison: temperature trajectories, hourly cost, discomfort scores, cumulative reward, and hour-by-hour AC decision timelines.

---

## Dynamic Weather Input

The app supports **live outdoor temperature data** via the [OpenWeatherMap](https://openweathermap.org) free API (1,000 calls/day):

1. Register at openweathermap.org → go to **API Keys** → copy your key.
2. In the app sidebar, select **Dynamic (API)** mode.
3. Type your city name and paste the API key, then click **Fetch Live Weather**.
4. The outdoor temperature profile is generated from the real current temperature for that city, keeping the natural diurnal shape of the default profile.

The **India residential time-of-use tariff** is used in both modes — no separate API is needed for pricing.

> **API key storage:** The key is entered in the sidebar each session. For persistence, you can save it in `.streamlit/secrets.toml`:
> ```toml
> OPENWEATHER_KEY = "your_key_here"
> ```
> Then read it in the app with `st.secrets["OPENWEATHER_KEY"]`.

---

## Project Structure

```
smart-building/
├── app.py            # Streamlit dashboard — simulation, charts, UI
├── env.py            # BuildingEnv class — physics, reward, state space
├── train.py          # Q-Learning training script
├── q_table.npy       # Trained Q-table (generated by train.py)
├── rewards_log.npy   # Per-episode rewards for the convergence plot
└── requirements.txt  # Python dependencies
```

---

## Team

**BCSE432E — Reinforcement Learning, VIT Vellore**

| Name |
|------|
| Hitakshi Sardana |
| Vaishali Chitipothu |
| Srivatsa Singaraju |
| Kush Agrawal |
