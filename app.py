"""
app.py  —  Smart Building RL · Energy Dashboard
Run  :  streamlit run app.py
Needs:  q_table.npy  (run python train.py first)
"""

import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False
from env import BuildingEnv, rule_based_action

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Building RL",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Vellore summer day: min ~25°C at 3–4 am, peak ~41°C at noon–1 pm
OUTDOOR_TEMPS = [27,26,26,25,25,26,28,31,34,37,39,40,41,41,40,39,37,35,33,31,30,29,28,27]
PRICE_PROFILE = [2,2,2,2,2,3,4,6,7,8,8,9,9,8,8,7,7,8,9,8,6,4,3,2]

# India residential time-of-use electricity tariff (₹/unit, same numeric scale as PRICE_PROFILE)
# Night off-peak (00-05): 3 | Morning/day (06-11): 5-6 | Afternoon (12-17): 6-7 | Evening peak (18-21): 9 | Taper (22-23): 6-4
INDIA_PRICE_PROFILE = [3,3,3,3,3,3, 5,6,6,6,6,6, 6,6,7,7,7,7, 9,9,9,8,6,4]


def _fetch_city_temp(city: str, api_key: str):
    """Return (current_temp_celsius, resolved_city_name).
    Raises on network or API error."""
    resp = _requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": api_key, "units": "metric"},
        timeout=6,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["main"]["temp"]), data["name"]


def _build_dynamic_outdoor_profile(current_temp: float, current_hour: int):
    """Shift the default 24-h profile to match the live current temperature,
    keeping the same diurnal shape."""
    offset = current_temp - OUTDOOR_TEMPS[current_hour]
    return [round(t + offset, 1) for t in OUTDOOR_TEMPS]

# ── design tokens ─────────────────────────────────────────────────────────────
BG        = "#0A0E1A"
SURFACE   = "#111827"
SURFACE2  = "#1C2333"
BORDER    = "#1E2D45"
ACCENT    = "#00D4FF"
ACCENT2   = "#FF6B35"
GREEN     = "#00E5A0"
RED       = "#FF4560"
MUTED     = "#4A6080"
TEXT      = "#E2EBF6"
TEXT2     = "#8BA4C0"
FONT_MONO = "'JetBrains Mono', 'Fira Code', monospace"
FONT_DISP = "'Barlow Condensed', 'Oswald', sans-serif"
FONT_BODY = "'DM Sans', 'Nunito', sans-serif"

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    background: {BG} !important;
    color: {TEXT};
    font-family: {FONT_BODY};
}}
#MainMenu, footer, header {{ visibility: hidden; }}

/* sidebar */
section[data-testid="stSidebar"] {{
    background: {SURFACE} !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

/* ── Slider fix: thumb + track ── */
section[data-testid="stSidebar"] .stSlider > div > div > div {{
    background: {ACCENT} !important;
}}
/* Slider min/max tick labels — remove cyan bg, just muted text */
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {{
    color: {MUTED} !important;
    font-size: 11px !important;
    font-family: {FONT_MONO} !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"] *,
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] * {{
    background: transparent !important;
    background-color: transparent !important;
    color: {MUTED} !important;
}}
/* Slider value bubble */
section[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumbValue"] {{
    color: {BG} !important;
    background: {ACCENT} !important;
    font-size: 11px !important;
    font-family: {FONT_MONO} !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
}}
/* Slider track background */
section[data-testid="stSidebar"] div[data-testid="stSlider"] > div {{
    background: {SURFACE2} !important;
}}

/* hide streamlit top bar */
div[data-testid="stToolbar"] {{ display: none; }}

/* metric cards */
.kpi-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin: 20px 0; }}
.kpi-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}}
.kpi-card.rl::before   {{ background: {ACCENT}; }}
.kpi-card.rule::before {{ background: {ACCENT2}; }}
.kpi-card.win::before  {{ background: {GREEN}; }}
.kpi-label {{ font-size: 11px; letter-spacing: .10em; text-transform: uppercase; color: {MUTED}; margin-bottom: 8px; }}
.kpi-value {{ font-family: {FONT_MONO}; font-size: 28px; font-weight: 500; color: {TEXT}; line-height: 1; }}
.kpi-sub   {{ font-size: 12px; color: {MUTED}; margin-top: 6px; }}
.kpi-delta {{ display:inline-block; font-size:12px; padding:3px 10px; border-radius:99px; margin-top:8px; font-weight:600; }}
.kpi-delta.good {{ background: rgba(0,229,160,.12); color:{GREEN}; }}
.kpi-delta.bad  {{ background: rgba(255,69,96,.12);  color:{RED}; }}
.kpi-delta.neu  {{ background: rgba(74,96,128,.15);  color:{MUTED}; }}

/* section labels */
.sec-label {{
    font-family: {FONT_DISP};
    font-size: 12px; letter-spacing: .16em; text-transform: uppercase;
    color: {TEXT2}; border-bottom: 1px solid {BORDER};
    padding-bottom: 6px; margin: 32px 0 16px;
    font-weight: 600;
}}

/* AC status cards — used via st.columns now */
.ac-card {{
    border-radius: 12px; padding: 20px 24px;
    display: flex; align-items: center; gap: 16px;
    border: 1px solid {BORDER};
    background: {SURFACE};
    min-height: 88px;
}}
.ac-card.on  {{ border-color: {ACCENT}; background: rgba(0,212,255,.05); }}
.ac-card.off {{ border-color: {BORDER}; }}
.ac-icon {{ font-size: 32px; line-height:1; }}
.ac-tag  {{ font-family:{FONT_MONO}; font-size: 16px; font-weight: 600; }}
.ac-tag.on  {{ color:{ACCENT}; }}
.ac-tag.off {{ color:{MUTED}; }}
.ac-info {{ font-size: 13px; color:{TEXT2}; margin-top:5px; }}
.ac-who  {{ font-size: 12px; color:{MUTED}; letter-spacing:.06em; text-transform:uppercase; margin-bottom: 4px; }}

/* hour timeline bar */
.timeline {{ display:flex; gap:3px; margin:12px 0; }}
.tbar {{
    flex:1; height:34px; border-radius:4px;
    display:flex; align-items:center; justify-content:center;
    font-size:9px; font-family:{FONT_MONO}; color:{BG}; font-weight:600;
    cursor:default;
}}
.tbar.on  {{ background:{ACCENT}; }}
.tbar.off {{ background:{SURFACE2}; color:{MUTED}; }}
.tbar.on2  {{ background:{ACCENT2}; }}

/* insight banner */
.insight {{
    border-radius:10px; padding:16px 20px;
    font-size: 14px; line-height:1.6;
    border-left: 3px solid;
    margin: 16px 0;
}}
.insight.win  {{ background:rgba(0,229,160,.07); border-color:{GREEN}; color:{GREEN}; }}
.insight.info {{ background:rgba(0,212,255,.07); border-color:{ACCENT}; color:{ACCENT}; }}
.insight.warn {{ background:rgba(255,107,53,.07); border-color:{ACCENT2}; color:{ACCENT2}; }}

/* table */
.stDataFrame {{ background: {SURFACE} !important; }}
div[data-testid="stDataFrameResizable"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden;
}}

/* sidebar inputs */
.stSelectbox > div > div {{ color: {TEXT} !important; }}
.stSelectbox > div > div > div {{ background: {SURFACE2} !important; color: {TEXT} !important; }}
div[data-baseweb="select"] > div {{ background: {SURFACE2} !important; border-color: {BORDER} !important; }}
div[data-baseweb="select"] span {{ color: {TEXT} !important; }}
div[data-baseweb="popover"] li {{ color: {TEXT} !important; background: {SURFACE} !important; }}
div[data-baseweb="popover"] li:hover {{ background: {SURFACE2} !important; }}
label {{ color: {TEXT2} !important; font-size: 13px !important; letter-spacing:.02em; }}
.stSlider label {{ color: {TEXT2} !important; }}

/* global font size floor — spans inherit from parent, not overridden here */
p {{ font-size: 13px; }}
div {{ font-size: 13px; }}
.stCaption {{ font-size: 12px !important; color: {MUTED} !important; }}

/* hero */
.hero {{
    padding: 36px 0 20px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 8px;
}}
.hero-title {{
    font-family: {FONT_DISP};
    font-size: 42px; font-weight: 800;
    letter-spacing: .03em; text-transform: uppercase;
    color: {TEXT}; line-height: 1;
    margin: 0;
}}
.hero-title span {{ color: {ACCENT}; font-size: inherit !important; font-weight: inherit !important; font-family: inherit !important; }}
.hero-sub {{ font-size: 14px; color:{MUTED}; margin-top:8px; letter-spacing:.04em; }}
.hero-badge {{
    display:inline-block; font-family:{FONT_MONO};
    font-size: 11px; padding:4px 12px;
    border: 1px solid {BORDER}; border-radius:99px;
    color:{MUTED}; margin-right:8px; margin-top:10px;
}}

/* plotly override */
.js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}

/* timeline legend text */
.tl-legend {{ font-size: 12px !important; color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def run_simulation(q_table, initial_temp, occupancy, outdoor_temps, price_profile):
    def _run(use_rl):
        env = BuildingEnv(
            initial_temp=initial_temp,
            initial_occupancy=occupancy,
            outdoor_temps=outdoor_temps,
            price_profile=price_profile,
        )
        env.reset()   # time is set to 0 in non-training mode
        state = env.get_state()
        done  = False
        rows  = []
        while not done:
            hour = env.time
            if use_rl:
                t, o, ti, p = state
                action = int(np.argmax(q_table[t, o, ti, p]))
            else:
                action = rule_based_action(env.indoor_temp)
            state, reward, done, energy, temp, cost, comfort = env.step(action)
            rows.append({
                "Hour": hour,
                "Temp": round(temp, 2),
                "Action": "ON" if action == 1 else "OFF",
                "Energy": round(energy, 2),
                "Cost": round(cost, 2),
                "Discomfort": round(comfort, 2),
                "Reward": round(reward, 2),
                "Price": round(env.price, 2),
            })
        return pd.DataFrame(rows)
    return _run(True), _run(False)


def plotly_cfg():
    return dict(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(family=FONT_BODY, color=TEXT2, size=13),
        margin        = dict(l=8, r=8, t=36, b=8),
        legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER,
                             borderwidth=1, font=dict(size=12)),
    )


def ax(title="", extra=None):
    d = dict(
        title=dict(text=title, font=dict(size=12, color=TEXT2)),
        gridcolor=BORDER,
        zerolinecolor=BORDER,
        tickfont=dict(size=11, family=FONT_MONO, color=TEXT2),
        linecolor=BORDER,
    )
    if extra:
        d.update(extra)
    return d


def delta_chip(rl_val, rule_val, lower_better=True):
    diff = rl_val - rule_val
    pct  = abs(diff / rule_val * 100) if rule_val else 0
    if lower_better:
        cls  = "good" if diff < 0 else ("bad" if diff > 0 else "neu")
        sign = "▼" if diff < 0 else ("▲" if diff > 0 else "—")
    else:
        cls  = "good" if diff > 0 else ("bad" if diff < 0 else "neu")
        sign = "▲" if diff > 0 else ("▼" if diff < 0 else "—")
    label = f"{sign} {pct:.1f}% vs rule"
    return f'<span class="kpi-delta {cls}">{label}</span>'


def _ac_card_html(controller, action, indoor_temp, outdoor_temp, note):
    """Render a single AC status card for a specific hour."""
    is_on      = action == "ON"
    card_css   = "on" if is_on else "off"
    icon       = "❄️" if is_on else "🌤️"
    tag_color  = ACCENT if is_on else MUTED
    return f"""
    <div class="ac-card {card_css}" style="margin-bottom:8px">
      <div class="ac-icon">{icon}</div>
      <div style="flex:1">
        <div class="ac-who">{controller}</div>
        <div class="ac-tag {card_css}" style="font-size:18px;font-weight:700">AC {action}</div>
        <div class="ac-info" style="margin-top:6px">
          Indoor (entering)&nbsp;<strong style="color:{TEXT}">{indoor_temp:.1f}°C</strong>
          &nbsp;·&nbsp;
          Outdoor&nbsp;<strong style="color:{MUTED}">{outdoor_temp}°C</strong>
        </div>
        <div style="font-size:11px;color:{MUTED};margin-top:4px;letter-spacing:.02em">{note}</div>
      </div>
    </div>"""


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 20px">
      <div style="font-family:{FONT_DISP};font-size:22px;font-weight:800;
                  letter-spacing:.05em;text-transform:uppercase;color:{TEXT}">
        Smart<span style="color:{ACCENT};font-size:inherit;font-weight:inherit">Building</span>
      </div>
      <div style="font-size:11px;letter-spacing:.12em;color:{MUTED};text-transform:uppercase;
                  margin-top:2px">RL Energy Controller · BCSE432E</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input mode toggle ─────────────────────────────────────────────────────
    st.markdown(f'<div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:{MUTED};margin-bottom:6px">Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "_mode",
        ["Manual", "Dynamic (API)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:12px 0"/>', unsafe_allow_html=True)

    # ── Manual mode ───────────────────────────────────────────────────────────
    if input_mode == "Manual":
        st.markdown(f'<div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:{MUTED};margin-bottom:8px">Simulation Parameters</div>', unsafe_allow_html=True)
        initial_temp      = st.slider("Indoor Start Temp (°C)", 20, 42, 30)
        outdoor_temps_sim = OUTDOOR_TEMPS
        price_profile_sim = INDIA_PRICE_PROFILE

    # ── Dynamic API mode ──────────────────────────────────────────────────────
    else:
        st.markdown(f'<div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:{MUTED};margin-bottom:8px">Live Weather (OpenWeatherMap)</div>', unsafe_allow_html=True)

        city    = st.text_input("City", value="Vellore",
                                placeholder="e.g. Mumbai, Chennai, Delhi")
        api_key = st.text_input("API Key", type="password",
                                help="Free key from openweathermap.org — register, go to API keys, copy it here.")

        fetch_btn = st.button("⬇  Fetch Live Weather", use_container_width=True,
                              disabled=not _REQUESTS_OK)
        if not _REQUESTS_OK:
            st.caption("Install `requests` to enable: `pip install requests`")

        if fetch_btn:
            if not api_key.strip():
                st.warning("Paste your OpenWeatherMap API key above first.")
            else:
                with st.spinner("Fetching…"):
                    try:
                        cur_temp, resolved = _fetch_city_temp(city.strip(), api_key.strip())
                        cur_hour           = datetime.datetime.now().hour
                        dyn_profile        = _build_dynamic_outdoor_profile(cur_temp, cur_hour)
                        st.session_state["dyn_outdoor"] = dyn_profile
                        st.session_state["dyn_city"]    = resolved
                        st.session_state["dyn_temp"]    = round(cur_temp, 1)
                    except Exception as exc:
                        st.error(f"Could not fetch: {exc}")

        if "dyn_outdoor" in st.session_state:
            st.markdown(
                f'<div style="font-size:12px;color:{GREEN};margin:6px 0 4px">'
                f'● {st.session_state["dyn_city"]} · {st.session_state["dyn_temp"]}°C live</div>',
                unsafe_allow_html=True,
            )
            outdoor_temps_sim = st.session_state["dyn_outdoor"]
        else:
            st.caption("Enter city + key and click Fetch to load real temperatures.")
            outdoor_temps_sim = OUTDOOR_TEMPS

        st.markdown("<br>", unsafe_allow_html=True)
        _default_temp = int(st.session_state.get("dyn_temp", 30))
        _default_temp = max(20, min(42, _default_temp))
        initial_temp      = st.slider("Indoor Start Temp (°C)", 20, 42, _default_temp)
        price_profile_sim = INDIA_PRICE_PROFILE

    # ── Occupancy (always shown) ───────────────────────────────────────────────
    occ_pct   = st.select_slider(
        "Occupancy",
        options=[0, 25, 50, 75, 100],
        value=100,
        format_func=lambda x: f"{x}%",
    )
    occupancy = occ_pct / 100.0   # float 0.0–1.0 passed to env

    st.markdown("<br>", unsafe_allow_html=True)

    q_exists = os.path.exists("q_table.npy")
    if q_exists:
        st.markdown(f'<div style="font-size:12px;color:{GREEN};margin-bottom:12px">● Q-table loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:12px;color:{RED};margin-bottom:12px">✕ q_table.npy not found — run python train.py</div>', unsafe_allow_html=True)

    run_btn = st.button("▶  Run Simulation", type="primary",
                        disabled=not q_exists, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px;color:{MUTED};line-height:2.2;letter-spacing:.02em;">
    <span style="font-size:12px;color:{TEXT2}">Team</span><br>
    Hitakshi Sardana · 23BAI0145<br>
    Vaishali Chitipothu · 23BAI0159<br>
    Srivatsa Singaraju · 23BAI0082<br>
    Kush Agrawal · 23BAI0024
    </div>
    """, unsafe_allow_html=True)


# ── hero ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1 class="hero-title">Intelligent <span>Energy</span> Management</h1>
  <div class="hero-sub">Reinforcement Learning vs Rule-Based Control · 24-Hour Simulation</div>
  <div style="margin-top:10px">
    <span class="hero-badge">Q-Learning</span>
    <span class="hero-badge">PPO-ready</span>
    <span class="hero-badge">HVAC + Lighting</span>
    <span class="hero-badge">BCSE432E</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not q_exists:
    st.markdown(f"""
    <div style="margin:48px auto;max-width:480px;text-align:center;padding:40px;
                background:{SURFACE};border:1px solid {BORDER};border-radius:16px;">
      <div style="font-size:36px;margin-bottom:16px">🏗️</div>
      <div style="font-family:{FONT_DISP};font-size:22px;font-weight:700;
                  text-transform:uppercase;color:{TEXT};margin-bottom:10px">
        Model Not Trained
      </div>
      <div style="font-size:13px;color:{MUTED};line-height:1.7">
        Run <code style="background:{SURFACE2};padding:2px 8px;border-radius:4px;
        color:{ACCENT};font-family:{FONT_MONO}">python train.py</code>
        in your terminal first, then reload this page.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

Q = np.load("q_table.npy")

# ── how it works (permanent collapsible) ─────────────────────────────────────
with st.expander("💡  How It Works", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "01", "Observe",
         f"The agent reads 4 inputs every hour: indoor temperature, occupancy level, time of day, and electricity price — forming one of 3,840 possible states.",
         ACCENT),
        (c2, "02", "Decide",
         f"The trained Q-table looks up the current state and selects the action (AC ON or OFF) with the highest expected future reward.",
         ACCENT),
        (c3, "03", "Act",
         f"The HVAC system executes the decision. AC ON cools the room by 1.2°C/hour; AC OFF lets the room exchange heat naturally with outdoors.",
         ACCENT2),
        (c4, "04", "Learn",
         f"During training, a reward signal (−cost − 2.5×discomfort) updated the Q-table via the Bellman equation across 100,000 episodes.",
         GREEN),
    ]
    for col, num, title, desc, color in cards:
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;
                        padding:18px 16px;position:relative;overflow:hidden;">
              <div style="font-family:{FONT_MONO};font-size:28px;font-weight:700;
                          color:{BORDER};position:absolute;top:8px;right:12px;line-height:1">{num}</div>
              <div style="font-family:{FONT_DISP};font-size:16px;font-weight:700;
                          text-transform:uppercase;color:{color};margin-bottom:8px">{title}</div>
              <div style="font-size:12px;color:{TEXT2};line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ── training curve expander ───────────────────────────────────────────────────
if os.path.exists("rewards_log.npy"):
    with st.expander("📈  Training Convergence Curve", expanded=False):
        rewards  = np.load("rewards_log.npy")
        smoothed = pd.Series(rewards).rolling(500).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=rewards, mode="lines", name="Episode reward",
            line=dict(color=BORDER, width=0.6), opacity=0.4,
        ))
        fig.add_trace(go.Scatter(
            y=smoothed, mode="lines", name="Rolling avg (500)",
            line=dict(color=ACCENT, width=2),
        ))
        fig.add_hline(y=float(np.mean(rewards[-2000:])),
                      line_dash="dot", line_color=GREEN, line_width=1,
                      annotation_text=f"Final avg {np.mean(rewards[-2000:]):.1f}",
                      annotation_font_color=GREEN, annotation_font_size=10)
        fig.update_layout(
            **plotly_cfg(),
            height=220,
            xaxis=ax("Episode"),
            yaxis=ax("Total reward"),
            title=dict(
                text="Q-Learning convergence — reward improves as the agent learns better cooling decisions (higher = better, closer to 0 = optimal)",
                font=dict(size=11, color=MUTED), x=0,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        c1, c2, c3 = st.columns(3)
        c1.metric("Episodes trained", f"{len(rewards):,}")
        c2.metric("Early avg (first 5k)",  f"{np.mean(rewards[:5000]):.1f}")
        c3.metric("Final avg (last 5k)", f"{np.mean(rewards[-5000:]):.1f}")

# ── run ──────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(""):
        df_rl, df_rule = run_simulation(
            Q, initial_temp, occupancy, outdoor_temps_sim, price_profile_sim,
        )
    st.session_state["df_rl"]           = df_rl
    st.session_state["df_rule"]         = df_rule
    st.session_state["outdoor_temps"]   = outdoor_temps_sim
    st.session_state["price_profile"]   = price_profile_sim

if "df_rl" not in st.session_state:
    st.markdown(f"""
    <div style="margin:56px auto;max-width:420px;text-align:center;padding:40px 32px;
                background:{SURFACE};border:1px solid {BORDER};border-radius:16px;">
      <div style="font-size:40px;margin-bottom:16px">🏢</div>
      <div style="font-family:{FONT_DISP};font-size:20px;font-weight:700;letter-spacing:.04em;
                  text-transform:uppercase;color:{TEXT};margin-bottom:10px">Ready to Simulate</div>
      <div style="font-size:13px;color:{MUTED};line-height:1.8">
        Configure parameters in the sidebar, then click
        <strong style="color:{ACCENT}">▶ Run Simulation</strong>
        to see the RL vs rule-based comparison.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df_rl              = st.session_state["df_rl"]
df_rule            = st.session_state["df_rule"]
outdoor_temps_disp = st.session_state.get("outdoor_temps", OUTDOOR_TEMPS)
price_profile_disp = st.session_state.get("price_profile", INDIA_PRICE_PROFILE)

# ── compute totals ────────────────────────────────────────────────────────────
rl_energy   = df_rl["Energy"].sum()
rl_cost     = df_rl["Cost"].sum()
rl_comfort  = df_rl["Discomfort"].sum()
rl_reward   = df_rl["Reward"].sum()
rl_ac_hrs   = (df_rl["Action"] == "ON").sum()

rule_energy  = df_rule["Energy"].sum()
rule_cost    = df_rule["Cost"].sum()
rule_comfort = df_rule["Discomfort"].sum()
rule_reward  = df_rule["Reward"].sum()
rule_ac_hrs  = (df_rule["Action"] == "ON").sum()

saved_cost    = rule_cost    - rl_cost
saved_comfort = rule_comfort - rl_comfort
saved_hrs     = rule_ac_hrs  - rl_ac_hrs

# ── insight banner ────────────────────────────────────────────────────────────
if rl_cost < rule_cost and rl_comfort < rule_comfort:
    ins_cls = "win"
    ins_msg = f"🏆  RL wins on both metrics — saved <strong>{saved_cost:.1f}</strong> in cost and <strong>{saved_comfort:.1f}</strong> discomfort units over 24 hours."
elif rl_cost < rule_cost:
    ins_cls = "info"
    ins_msg = f"💰  RL saved <strong>{saved_cost:.1f}</strong> in energy cost by cooling strategically during cheap-price windows."
elif rl_comfort < rule_comfort:
    ins_cls = "info"
    ins_msg = f"🌡  RL maintained better occupant comfort (−{saved_comfort:.1f} discomfort) at the cost of slightly higher energy spend."
else:
    ins_cls = "warn"
    ins_msg = "⚠️  Rule-based edged ahead this run — try a higher starting temperature or train for more episodes."

st.markdown(f'<div class="insight {ins_cls}">{ins_msg}</div>', unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-label">24-Hour Results at a Glance — RL vs Rule-Based</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-grid">

  <div class="kpi-card rl">
    <div class="kpi-label">⚡ Total Energy Consumed</div>
    <div class="kpi-value">{rl_energy:.1f}<span style="font-size:14px;color:{MUTED}"> kWh</span></div>
    <div class="kpi-sub">RL used {rl_energy:.1f} kWh &nbsp;·&nbsp; Rule-based used {rule_energy:.1f} kWh</div>
    {delta_chip(rl_energy, rule_energy, lower_better=True)}
  </div>

  <div class="kpi-card rl">
    <div class="kpi-label">💰 Total Electricity Cost (₹)</div>
    <div class="kpi-value">₹{rl_cost:.1f}</div>
    <div class="kpi-sub">RL spent ₹{rl_cost:.1f} &nbsp;·&nbsp; Rule-based spent ₹{rule_cost:.1f}</div>
    {delta_chip(rl_cost, rule_cost, lower_better=True)}
  </div>

  <div class="kpi-card rl">
    <div class="kpi-label">😰 Total Comfort Penalty</div>
    <div class="kpi-value">{rl_comfort:.1f}</div>
    <div class="kpi-sub">Lower is better &nbsp;·&nbsp; Rule-based scored {rule_comfort:.1f}</div>
    {delta_chip(rl_comfort, rule_comfort, lower_better=True)}
  </div>

</div>
""", unsafe_allow_html=True)

# ── AC status at critical moments ─────────────────────────────────────────────
st.markdown(
    f'<div class="sec-label">HVAC Decisions at Critical Moments — What Each Controller Actually Did</div>',
    unsafe_allow_html=True,
)

# Hour 13 = peak outdoor heat | Hour 19 = peak electricity price (₹9/unit)
# Use temp at end of previous hour as "entering temp" — this is what the controller
# saw when making its decision, not the post-action result stored in iloc[13/19].
_h13_rl_a   = df_rl["Action"].iloc[13];   _h13_rl_t   = df_rl["Temp"].iloc[12]
_h13_rule_a = df_rule["Action"].iloc[13]; _h13_rule_t = df_rule["Temp"].iloc[12]
_h19_rl_a   = df_rl["Action"].iloc[19];   _h19_rl_t   = df_rl["Temp"].iloc[18]
_h19_rule_a = df_rule["Action"].iloc[19]; _h19_rule_t = df_rule["Temp"].iloc[18]
_out13      = outdoor_temps_disp[13]
_out19      = outdoor_temps_disp[19]
_pr19       = price_profile_disp[19]

col_h, col_p = st.columns(2)

with col_h:
    st.markdown(
        f'<div style="font-size:12px;font-weight:700;color:{TEXT2};letter-spacing:.06em;'
        f'text-transform:uppercase;margin-bottom:10px">'
        f'🌡️ Hour 13 · 1 pm · Outdoor {_out13}°C (Hottest Point)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        _ac_card_html(
            "🤖 RL Controller", _h13_rl_a, _h13_rl_t, _out13,
            "RL weighs cost + comfort — may pre-cool before this hour",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _ac_card_html(
            "📏 Rule-Based", _h13_rule_a, _h13_rule_t, _out13,
            "Rule: turn ON if indoor temp > 25°C, regardless of price",
        ),
        unsafe_allow_html=True,
    )

with col_p:
    st.markdown(
        f'<div style="font-size:12px;font-weight:700;color:{TEXT2};letter-spacing:.06em;'
        f'text-transform:uppercase;margin-bottom:10px">'
        f'⚡ Hour 19 · 7 pm · ₹{_pr19}/unit (Peak Tariff)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        _ac_card_html(
            "🤖 RL Controller", _h19_rl_a, _h19_rl_t, _out19,
            "RL avoids running AC at ₹9/unit when possible",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        _ac_card_html(
            "📏 Rule-Based", _h19_rule_a, _h19_rule_t, _out19,
            "Rule: unaware of price — runs AC if temp > 25°C",
        ),
        unsafe_allow_html=True,
    )

# ── hourly AC timeline bars ───────────────────────────────────────────────────
st.markdown(f'<div class="sec-label">Hour-by-Hour AC Decisions — Cyan = RL ON · Orange = Rule ON · Grey = OFF</div>', unsafe_allow_html=True)

def timeline_html(actions, color_class):
    bars = ""
    for i, a in enumerate(actions):
        css = color_class if a == "ON" else "off"
        label = str(i) if i % 4 == 0 else ("ON" if a == "ON" else "")
        bars += f'<div class="tbar {css}" title="Hour {i}: AC {a}">{label}</div>'
    return f'<div class="timeline">{bars}</div>'

st.markdown(f"""
<div style="display:grid;grid-template-columns:80px 1fr;gap:8px;align-items:center;margin-bottom:4px">
  <div style="font-size:12px;color:{ACCENT};text-align:right;padding-right:10px;font-weight:600">🤖 RL</div>
  <div>{timeline_html(df_rl["Action"].tolist(), "on")}</div>
</div>
<div style="display:grid;grid-template-columns:80px 1fr;gap:8px;align-items:center">
  <div style="font-size:12px;color:{ACCENT2};text-align:right;padding-right:10px;font-weight:600">📏 Rule</div>
  <div>{timeline_html(df_rule["Action"].tolist(), "on2")}</div>
</div>
<div style="display:flex;gap:20px;margin-top:10px;font-size:12px;color:{MUTED};flex-wrap:wrap;align-items:center">
  <span><span style="color:{ACCENT}">■</span>&nbsp;RL AC ON</span>
  <span><span style="color:{ACCENT2}">■</span>&nbsp;Rule AC ON</span>
  <span><span style="color:{SURFACE2}">■</span>&nbsp;AC OFF</span>
  <span style="margin-left:4px;color:{TEXT2}">
    RL ran AC for <strong style="color:{ACCENT}">{rl_ac_hrs} hours</strong> out of 24 &nbsp;·&nbsp;
    Rule-based ran for <strong style="color:{ACCENT2}">{rule_ac_hrs} hours</strong> out of 24 &nbsp;·&nbsp;
    <strong style="color:{GREEN}">RL used {abs(saved_hrs)} fewer cooling hours</strong>
  </span>
</div>
""", unsafe_allow_html=True)

# ── temperature chart ─────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-label">Indoor Temperature Over 24 Hours — Does RL Keep the Room Cooler?</div>', unsafe_allow_html=True)

hours = list(range(24))
fig_temp = go.Figure()

# Comfort band
fig_temp.add_hrect(
    y0=22, y1=24, fillcolor=GREEN, opacity=0.08, line_width=0,
    annotation_text="Comfort zone 22–24°C",
    annotation_position="top right",
    annotation_font=dict(size=10, color=GREEN),
)

# Outdoor temperature reference
fig_temp.add_trace(go.Scatter(
    x=hours, y=outdoor_temps_disp, name="Outdoor Temp",
    mode="lines", line=dict(color=MUTED, width=1.5, dash="dot"),
    fill="tozeroy", fillcolor="rgba(74,96,128,0.04)",
))
# Rule-based indoor temperature
fig_temp.add_trace(go.Scatter(
    x=hours, y=df_rule["Temp"].tolist(), name="Rule-Based Indoor",
    mode="lines", line=dict(color=ACCENT2, width=2.5),
))
# RL indoor temperature
fig_temp.add_trace(go.Scatter(
    x=hours, y=df_rl["Temp"].tolist(), name="RL Indoor",
    mode="lines", line=dict(color=ACCENT, width=2.5),
))

# Mark hours when RL had AC ON (triangle down = cooling event)
ac_on_rl   = [h for h, a in zip(hours, df_rl["Action"])   if a == "ON"]
ac_on_rule = [h for h, a in zip(hours, df_rule["Action"]) if a == "ON"]
if ac_on_rl:
    fig_temp.add_trace(go.Scatter(
        x=ac_on_rl, y=[df_rl["Temp"].iloc[h] for h in ac_on_rl],
        name="RL cooling active",
        mode="markers", marker=dict(symbol="triangle-down", size=11, color=ACCENT, opacity=0.9),
    ))
if ac_on_rule:
    fig_temp.add_trace(go.Scatter(
        x=ac_on_rule, y=[df_rule["Temp"].iloc[h] for h in ac_on_rule],
        name="Rule cooling active",
        mode="markers", marker=dict(symbol="triangle-down", size=11, color=ACCENT2, opacity=0.9),
    ))

fig_temp.update_layout(
    **plotly_cfg(), height=360,
    xaxis=ax("Hour of day", {"tickvals": list(range(0, 24, 2))}),
    yaxis=ax("Temperature (°C)", {"range": [15, 46]}),
    title=dict(
        text="Green band = comfort zone. Triangles show when AC was active. Vellore peak outdoor: 41°C.",
        font=dict(size=11, color=MUTED), x=0,
    ),
)
st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})

# ── cost vs electricity price (merged) ───────────────────────────────────────
st.markdown(f'<div class="sec-label">Electricity Spend Per Hour — Does RL Avoid the Expensive Hours?</div>', unsafe_allow_html=True)

fig_cost = make_subplots(specs=[[{"secondary_y": True}]])

fig_cost.add_trace(go.Bar(
    x=hours, y=df_rule["Cost"].tolist(), name="Rule-Based Cost",
    marker_color=ACCENT2, opacity=0.7,
), secondary_y=False)
fig_cost.add_trace(go.Bar(
    x=hours, y=df_rl["Cost"].tolist(), name="RL Cost",
    marker_color=ACCENT, opacity=0.9,
), secondary_y=False)
fig_cost.add_trace(go.Scatter(
    x=hours, y=price_profile_disp, name="Electricity Price (₹/unit)",
    mode="lines", line=dict(color=RED, width=2, dash="dot"),
), secondary_y=True)

peak_hrs  = [h for h, p in enumerate(price_profile_disp) if p >= 8]
peak_vals = [price_profile_disp[h] for h in peak_hrs]
fig_cost.add_trace(go.Scatter(
    x=peak_hrs, y=peak_vals, name="Peak hours",
    mode="markers", marker=dict(color=RED, size=8, symbol="circle"),
    showlegend=False,
), secondary_y=True)

_base = plotly_cfg()
fig_cost.update_layout(
    **_base, height=300, barmode="group",
    xaxis =ax("Hour of day", {"tickvals": list(range(0, 24, 2))}),
    title=dict(
        text="Bars = ₹ spent each hour (cyan = RL, orange = Rule-based)  ·  Dotted line = tariff rate  ·  Red dots = peak ₹9/unit hours",
        font=dict(size=11, color=MUTED), x=0,
    ),
)
fig_cost.update_yaxes(
    title_text="Cost (₹)", secondary_y=False,
    gridcolor=BORDER, tickfont=dict(size=11, family=FONT_MONO, color=TEXT2),
    title_font=dict(size=11, color=TEXT2),
)
fig_cost.update_yaxes(
    title_text="Price (₹/unit)", secondary_y=True,
    showgrid=False, tickfont=dict(size=11, family=FONT_MONO, color=TEXT2),
    title_font=dict(size=11, color=TEXT2),
)
st.plotly_chart(fig_cost, use_container_width=True, config={"displayModeBar": False})

# ── raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋  Raw Hourly Data"):
    t1, t2 = st.tabs(["🤖 RL Controller", "📏 Rule-Based"])
    with t1:
        st.dataframe(df_rl.set_index("Hour"), use_container_width=True)
    with t2:
        st.dataframe(df_rule.set_index("Hour"), use_container_width=True)