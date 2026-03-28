"""
app.py  —  Smart Building RL · Energy Dashboard
Run  :  streamlit run app.py
Needs:  q_table.npy  (run python train.py first)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from env import BuildingEnv, rule_based_action

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Building RL",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTDOOR_TEMPS = [22,21,21,20,20,21,23,26,28,30,32,33,34,34,33,32,31,30,28,27,26,25,24,23]
PRICE_PROFILE = [2,2,2,2,2,3,4,6,7,8,8,9,9,8,8,7,7,8,9,8,6,4,3,2]

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
.kpi-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 20px 0; }}
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
def run_simulation(q_table, initial_temp, initial_price, occupancy):
    def _run(use_rl):
        env = BuildingEnv(initial_temp=initial_temp, initial_occupancy=occupancy)
        env.reset()
        env.price = initial_price
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

    st.markdown(f'<div style="font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:{MUTED};margin-bottom:8px">Simulation Parameters</div>', unsafe_allow_html=True)

    initial_temp  = st.slider("Indoor Start Temp (°C)", 18, 38, 28)
    initial_price = st.slider("Starting Energy Price",   2,  9,  5)
    occupancy     = st.selectbox("Occupancy",
                                 [1, 0],
                                 format_func=lambda x: "Occupied" if x == 1 else "Empty")

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
            title=dict(text="Agent learns over time — reward rises toward 0",
                       font=dict(size=12, color=MUTED), x=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        c1, c2, c3 = st.columns(3)
        c1.metric("Episodes trained", f"{len(rewards):,}")
        c2.metric("Early avg (first 5k)",  f"{np.mean(rewards[:5000]):.1f}")
        c3.metric("Final avg (last 5k)", f"{np.mean(rewards[-5000:]):.1f}")

# ── run ──────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner(""):
        df_rl, df_rule = run_simulation(Q, initial_temp, initial_price, occupancy)
    st.session_state["df_rl"]   = df_rl
    st.session_state["df_rule"] = df_rule

if "df_rl" not in st.session_state:
    st.markdown(f'<div class="sec-label">How it works</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "01", "Observe", f"Agent reads temp, occupancy, time of day, and energy price — 3,840 possible states", ACCENT),
        (c2, "02", "Decide",  f"Q-table selects the action (AC ON / OFF) with the highest expected future reward",    ACCENT),
        (c3, "03", "Act",     f"HVAC is controlled — room temperature and energy consumption update accordingly",     ACCENT2),
        (c4, "04", "Learn",   f"Reward signal (cost + discomfort) updates the Q-table via the Bellman equation",      GREEN),
    ]
    for col, num, title, desc, color in cards:
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;
                        padding:20px 18px;height:180px;position:relative;overflow:hidden;">
              <div style="font-family:{FONT_MONO};font-size:36px;font-weight:700;
                          color:{BORDER};position:absolute;top:10px;right:14px;line-height:1">{num}</div>
              <div style="font-family:{FONT_DISP};font-size:18px;font-weight:700;
                          text-transform:uppercase;color:{color};margin-bottom:10px">{title}</div>
              <div style="font-size:13px;color:{TEXT2};line-height:1.6">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

df_rl   = st.session_state["df_rl"]
df_rule = st.session_state["df_rule"]

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
st.markdown(f'<div class="sec-label">Performance Summary</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-grid">

  <div class="kpi-card rl">
    <div class="kpi-label">⚡ Energy Used · RL</div>
    <div class="kpi-value">{rl_energy:.1f}<span style="font-size:14px;color:{MUTED}"> kWh</span></div>
    <div class="kpi-sub">Rule-based: {rule_energy:.1f} kWh</div>
    {delta_chip(rl_energy, rule_energy, lower_better=True)}
  </div>

  <div class="kpi-card rl">
    <div class="kpi-label">💰 Total Cost · RL</div>
    <div class="kpi-value">{rl_cost:.1f}</div>
    <div class="kpi-sub">Rule-based: {rule_cost:.1f}</div>
    {delta_chip(rl_cost, rule_cost, lower_better=True)}
  </div>

  <div class="kpi-card rl">
    <div class="kpi-label">😰 Discomfort Score · RL</div>
    <div class="kpi-value">{rl_comfort:.1f}</div>
    <div class="kpi-sub">Rule-based: {rule_comfort:.1f}</div>
    {delta_chip(rl_comfort, rule_comfort, lower_better=True)}
  </div>

  <div class="kpi-card win">
    <div class="kpi-label">🏆 Total Reward · RL</div>
    <div class="kpi-value">{rl_reward:.1f}</div>
    <div class="kpi-sub">Rule-based: {rule_reward:.1f}</div>
    {delta_chip(rl_reward, rule_reward, lower_better=False)}
  </div>

</div>
""", unsafe_allow_html=True)

# ── AC status ─────────────────────────────────────────────────────────────────
# FIX: use st.columns instead of a single st.markdown with nested HTML —
# that caused Streamlit to render the raw HTML as a code block.
st.markdown(f'<div class="sec-label">Final Hour AC Status</div>', unsafe_allow_html=True)

final_rl_action   = df_rl["Action"].iloc[-1]
final_rule_action = df_rule["Action"].iloc[-1]
final_rl_temp     = df_rl["Temp"].iloc[-1]
final_rule_temp   = df_rule["Temp"].iloc[-1]

col_rl, col_rule = st.columns(2)

with col_rl:
    rl_css  = "on" if final_rl_action == "ON" else "off"
    rl_icon = "❄️" if final_rl_action == "ON" else "🌤️"
    st.markdown(f"""
    <div class="ac-card {rl_css}">
      <div class="ac-icon">{rl_icon}</div>
      <div>
        <div class="ac-who">🤖 RL Controller</div>
        <div class="ac-tag {rl_css}">AC {final_rl_action}</div>
        <div class="ac-info">Indoor temp: <strong style="color:{TEXT}">{final_rl_temp}°C</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_rule:
    rule_css  = "on" if final_rule_action == "ON" else "off"
    rule_icon = "❄️" if final_rule_action == "ON" else "🌤️"
    st.markdown(f"""
    <div class="ac-card {rule_css}">
      <div class="ac-icon">{rule_icon}</div>
      <div>
        <div class="ac-who">📏 Rule-Based</div>
        <div class="ac-tag {rule_css}">AC {final_rule_action}</div>
        <div class="ac-info">Indoor temp: <strong style="color:{TEXT}">{final_rule_temp}°C</strong></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── hourly AC timeline bars ───────────────────────────────────────────────────
st.markdown(f'<div class="sec-label">Hourly AC Decisions · 24 Hours</div>', unsafe_allow_html=True)

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
  <span style="margin-left:4px;color:{TEXT2}">RL ON: <strong>{rl_ac_hrs}h</strong> / 24 &nbsp;·&nbsp; Rule ON: <strong>{rule_ac_hrs}h</strong> / 24 &nbsp;·&nbsp; RL saved <strong>{abs(saved_hrs)}h</strong> of cooling</span>
</div>
""", unsafe_allow_html=True)

# ── temperature chart ─────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-label">Temperature Comparison · Indoor vs Outdoor</div>', unsafe_allow_html=True)

hours = list(range(24))
fig_temp = go.Figure()

fig_temp.add_hrect(y0=22, y1=24, fillcolor=GREEN, opacity=0.06,
                   line_width=0, annotation_text="Comfort 22–24°C",
                   annotation_position="top right",
                   annotation_font=dict(size=10, color=GREEN))

fig_temp.add_trace(go.Scatter(
    x=hours, y=OUTDOOR_TEMPS, name="Outdoor",
    mode="lines", line=dict(color=MUTED, width=1.5, dash="dot"),
    fill="tozeroy", fillcolor=f"rgba(74,96,128,0.06)",
))
fig_temp.add_trace(go.Scatter(
    x=hours, y=df_rule["Temp"].tolist(), name="Rule-Based",
    mode="lines+markers", line=dict(color=ACCENT2, width=2),
    marker=dict(size=5, color=ACCENT2),
))
fig_temp.add_trace(go.Scatter(
    x=hours, y=df_rl["Temp"].tolist(), name="RL Controller",
    mode="lines+markers", line=dict(color=ACCENT, width=2.5),
    marker=dict(size=5, color=ACCENT),
))

ac_on_hours = [h for h, a in zip(hours, df_rl["Action"]) if a == "ON"]
ac_on_temps = [df_rl["Temp"].iloc[h] for h in ac_on_hours]
fig_temp.add_trace(go.Scatter(
    x=ac_on_hours, y=ac_on_temps, name="RL AC ON",
    mode="markers", marker=dict(symbol="triangle-down", size=10, color=ACCENT, opacity=0.8),
))

fig_temp.update_layout(
    **plotly_cfg(), height=320,
    xaxis=ax("Hour of day", {"tickvals": list(range(0,24,2))}),
    yaxis=ax("Temperature (°C)"),
)
st.plotly_chart(fig_temp, use_container_width=True, config={"displayModeBar": False})

# ── cost + price side by side ─────────────────────────────────────────────────
st.markdown(f'<div class="sec-label">Cost & Energy Price · Hourly</div>', unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Bar(
        x=hours, y=df_rule["Cost"].tolist(), name="Rule Cost",
        marker_color=ACCENT2, opacity=0.7,
    ))
    fig_cost.add_trace(go.Bar(
        x=hours, y=df_rl["Cost"].tolist(), name="RL Cost",
        marker_color=ACCENT, opacity=0.9,
    ))
    fig_cost.update_layout(
        **plotly_cfg(), height=240, barmode="group",
        xaxis=ax("Hour", {"tickvals": list(range(0,24,2))}),
        yaxis=ax("Cost"),
        title=dict(text="Hourly cost comparison", font=dict(size=12, color=MUTED), x=0),
    )
    st.plotly_chart(fig_cost, use_container_width=True, config={"displayModeBar": False})

with col_r:
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=hours, y=PRICE_PROFILE, name="Energy Price",
        mode="lines", line=dict(color=ACCENT2, width=2),
        fill="tozeroy", fillcolor=f"rgba(255,107,53,0.10)",
    ))
    peak_hrs = [h for h, p in enumerate(PRICE_PROFILE) if p >= 8]
    peak_vals = [PRICE_PROFILE[h] for h in peak_hrs]
    fig_price.add_trace(go.Scatter(
        x=peak_hrs, y=peak_vals, name="Peak price",
        mode="markers", marker=dict(color=RED, size=7, symbol="circle"),
    ))
    fig_price.update_layout(
        **plotly_cfg(), height=240,
        xaxis=ax("Hour", {"tickvals": list(range(0,24,2))}),
        yaxis=ax("Price per unit"),
        title=dict(text="Time-of-use electricity price — RL avoids peaks", font=dict(size=12, color=MUTED), x=0),
    )
    st.plotly_chart(fig_price, use_container_width=True, config={"displayModeBar": False})

# ── cumulative reward ─────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-label">Cumulative Reward · RL vs Rule-Based</div>', unsafe_allow_html=True)

fig_rew = go.Figure()
fig_rew.add_trace(go.Scatter(
    x=hours, y=df_rule["Reward"].cumsum().tolist(), name="Rule-Based",
    mode="lines", line=dict(color=ACCENT2, width=2, dash="dash"),
    fill="tozeroy", fillcolor="rgba(255,107,53,0.05)",
))
fig_rew.add_trace(go.Scatter(
    x=hours, y=df_rl["Reward"].cumsum().tolist(), name="RL Controller",
    mode="lines", line=dict(color=ACCENT, width=2.5),
    fill="tozeroy", fillcolor="rgba(0,212,255,0.07)",
))
fig_rew.add_hline(y=0, line_color=MUTED, line_width=0.8, line_dash="dot")
fig_rew.update_layout(
    **plotly_cfg(), height=260,
    xaxis=ax("Hour of day", {"tickvals": list(range(0,24,2))}),
    yaxis=ax("Cumulative reward"),
    annotations=[dict(x=23, y=df_rl["Reward"].cumsum().iloc[-1],
                      text=f"RL: {df_rl['Reward'].cumsum().iloc[-1]:.1f}",
                      showarrow=False, font=dict(color=ACCENT, size=11), xanchor="right")],
)
st.plotly_chart(fig_rew, use_container_width=True, config={"displayModeBar": False})
st.caption("Reward is always negative — the agent tries to get as close to 0 as possible. Higher = better.")

# ── discomfort ────────────────────────────────────────────────────────────────
st.markdown(f'<div class="sec-label">Discomfort Score · Hourly</div>', unsafe_allow_html=True)

fig_dis = go.Figure()
fig_dis.add_trace(go.Bar(
    x=hours, y=df_rule["Discomfort"].tolist(), name="Rule-Based",
    marker_color=ACCENT2, opacity=0.6,
))
fig_dis.add_trace(go.Bar(
    x=hours, y=df_rl["Discomfort"].tolist(), name="RL Controller",
    marker_color=ACCENT, opacity=0.85,
))
fig_dis.update_layout(
    **plotly_cfg(), height=220, barmode="group",
    xaxis=ax("Hour", {"tickvals": list(range(0,24,2))}),
    yaxis=ax("Discomfort units"),
    title=dict(text="0 = perfectly comfortable (22–24°C). Higher = worse.",
               font=dict(size=12, color=MUTED), x=0),
)
st.plotly_chart(fig_dis, use_container_width=True, config={"displayModeBar": False})

# ── raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋  Raw Hourly Data"):
    t1, t2 = st.tabs(["🤖 RL Controller", "📏 Rule-Based"])
    with t1:
        st.dataframe(df_rl.set_index("Hour"), use_container_width=True)
    with t2:
        st.dataframe(df_rule.set_index("Hour"), use_container_width=True)