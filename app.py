"""Streamlit match-day dashboard for the saved IPL score regression pipeline."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ipl_score_model.pkl"
HERO_IMAGE = ROOT / "assets" / "ipl_stadium.jpg"

st.set_page_config(
    page_title="IPL First Innings Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifact():
    """Load the fitted pipeline once; it contains its own preprocessing steps."""
    return joblib.load(MODEL_PATH)


def inject_styles() -> None:
    """Set a polished dark match-night theme with restrained CSS motion."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap');
        :root {
          --ink: #f4f7fb; --muted: #9ba8bd; --line: rgba(255,255,255,.10);
          --panel: rgba(13,23,40,.84); --lime: #c9f36a; --cyan: #54d8df;
        }
        .stApp { background: radial-gradient(ellipse at 85% 0%, #132b3e 0%, #091321 42%, #07101c 100%); color: var(--ink); }
        .main .block-container { max-width: 1420px; padding-top: 1.45rem; padding-bottom: 3rem; }
        h1, h2, h3, p, label, [data-testid="stMetricLabel"] { font-family: 'DM Sans', sans-serif; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#0c1727,#0a1220); border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label { color: #f4f7fb; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #c3cedc; font-size: .9rem; }
        [data-testid="stMetric"] { background: rgba(17,31,50,.82); border: 1px solid var(--line); padding: 1.05rem 1.1rem; border-radius: 16px; }
        [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg,var(--cyan),var(--lime)); }
        div[data-testid="stExpander"] { background: rgba(15,26,43,.65); border: 1px solid var(--line); border-radius: 14px; }
        .hero {
          position: relative; overflow: hidden; min-height: 285px; padding: 2.25rem 2.55rem;
          border: 1px solid rgba(255,255,255,.15); border-radius: 24px; margin-bottom: 1.55rem;
          display: flex; align-items: center; isolation: isolate; background-position: center 45%; background-size: cover;
          box-shadow: 0 22px 60px rgba(0,0,0,.3); animation: rise-in .7s ease both;
        }
        .hero::before { content:''; position:absolute; inset:0; z-index:-1; background:linear-gradient(90deg,rgba(5,12,23,.96) 0%,rgba(5,12,23,.79) 43%,rgba(5,12,23,.25) 100%),linear-gradient(0deg,rgba(5,12,23,.45),transparent 60%); }
        .hero::after { content:''; position:absolute; inset:-35%; z-index:-1; opacity:.3; background:linear-gradient(115deg,transparent 38%,rgba(201,243,106,.18) 50%,transparent 62%); transform:translateX(-55%); animation: stadium-sheen 8s ease-in-out infinite; }
        .hero-copy { max-width: 780px; }
        .eyebrow { color:var(--lime); letter-spacing:.19em; text-transform:uppercase; font-size:.73rem; font-weight:800; margin:0 0 .72rem; }
        .hero h1 { font-family:'Barlow Condensed',sans-serif; font-size:clamp(2.65rem,5vw,4.65rem); line-height:.95; letter-spacing:.015em; text-transform:uppercase; margin:.1rem 0 .8rem; color:white; }
        .hero h1 span { color:var(--lime); }
        .hero p { color:#d1d9e5; max-width:610px; font-size:1rem; margin:0; }
        .hero-tag { display:inline-flex; align-items:center; gap:.55rem; border:1px solid rgba(255,255,255,.22); background:rgba(4,12,22,.42); border-radius:99px; padding:.46rem .8rem; color:#e5edf6; font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:1.1rem; }
        .live-dot { width:8px; height:8px; border-radius:50%; background:#c9f36a; box-shadow:0 0 0 0 rgba(201,243,106,.55); animation:live-pulse 1.8s ease-out infinite; }
        .section-kicker { color:var(--lime); letter-spacing:.15em; font-size:.7rem; font-weight:800; text-transform:uppercase; margin-bottom:.2rem; }
        .matchup { display:flex; align-items:stretch; gap:.8rem; margin:.35rem 0 1rem; animation:rise-in .8s .12s ease both; }
        .team-card { flex:1; padding:.95rem 1.1rem; border:1px solid var(--line); border-radius:15px; background:linear-gradient(135deg,rgba(20,38,59,.95),rgba(12,22,37,.88)); }
        .team-card.bat { border-left:3px solid var(--lime); }
        .team-card.bowl { border-left:3px solid var(--cyan); }
        .team-card small { display:block; color:var(--muted); font-weight:700; font-size:.67rem; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.35rem; }
        .team-card strong { font-family:'Barlow Condensed',sans-serif; font-size:1.48rem; color:#fff; }
        .versus { display:flex; align-items:center; color:#718096; font-family:'Barlow Condensed',sans-serif; font-size:1rem; font-weight:800; }
        .score-panel { min-height:245px; padding:1.45rem 1.6rem; border-radius:20px; border:1px solid rgba(201,243,106,.27); background:radial-gradient(ellipse at 100% 0%,rgba(201,243,106,.12),transparent 55%),linear-gradient(135deg,rgba(17,32,50,.98),rgba(10,20,34,.96)); box-shadow:0 18px 45px rgba(0,0,0,.2); animation:rise-in .8s .2s ease both; }
        .score-label { color:#afbbc9; font-weight:700; letter-spacing:.13em; font-size:.72rem; text-transform:uppercase; }
        .score-value { font-family:'Barlow Condensed',sans-serif; color:var(--lime); font-weight:800; font-size:clamp(4rem,7vw,6rem); line-height:1; margin:.25rem 0 .12rem; text-shadow:0 0 32px rgba(201,243,106,.2); animation:score-glow 3.4s ease-in-out infinite; }
        .score-unit { color:#dce5ee; font-size:1rem; font-weight:600; }
        .score-note { color:#9eacbd; margin:.75rem 0 0; font-size:.88rem; }
        .mini-stat { display:inline-flex; align-items:center; gap:.35rem; border-radius:8px; padding:.42rem .62rem; margin:.85rem .35rem 0 0; background:rgba(255,255,255,.06); color:#dce5ee; font-size:.78rem; }
        .mini-stat b { color:white; }
        .insight-title { font-family:'Barlow Condensed',sans-serif; text-transform:uppercase; letter-spacing:.035em; font-size:1.7rem; color:#fff; margin-bottom:.35rem; }
        .stCaption, [data-testid="stCaptionContainer"] { color:#91a1b5; }
        .footer-line { border-top:1px solid var(--line); margin-top:1.7rem; padding-top:.9rem; color:#8291a5; font-size:.75rem; }
        @keyframes rise-in { from { opacity:0; transform:translateY(14px) } to { opacity:1; transform:translateY(0) } }
        @keyframes live-pulse { 0% { box-shadow:0 0 0 0 rgba(201,243,106,.55) } 70% { box-shadow:0 0 0 8px rgba(201,243,106,0) } 100% { box-shadow:0 0 0 0 rgba(201,243,106,0) } }
        @keyframes stadium-sheen { 0%,40% { transform:translateX(-55%) } 75%,100% { transform:translateX(55%) } }
        @keyframes score-glow { 0%,100% { text-shadow:0 0 22px rgba(201,243,106,.12) } 50% { text-shadow:0 0 40px rgba(201,243,106,.34) } }
        @media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; } }
        @media (max-width: 700px) { .hero { min-height:245px; padding:1.4rem; } .matchup { gap:.45rem; } .team-card { padding:.75rem; } .team-card strong { font-size:1.15rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

if not MODEL_PATH.exists():
    st.error("The trained model is not available yet.")
    st.code("python setup_data.py\npython train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

artifact = load_artifact()
teams = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
    "Kolkata Knight Riders", "Rajasthan Royals", "Sunrisers Hyderabad",
    "Delhi Capitals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants",
]
venues = [
    "Wankhede Stadium", "M. A. Chidambaram Stadium", "Eden Gardens",
    "M. Chinnaswamy Stadium", "Narendra Modi Stadium", "Arun Jaitley Stadium",
    "Rajiv Gandhi International Stadium", "Sawai Mansingh Stadium",
    "Punjab Cricket Association Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
]

with st.sidebar:
    st.markdown("<div class='section-kicker'>Set the scene</div>", unsafe_allow_html=True)
    st.header("Match situation")
    batting_team = st.selectbox("Batting team", teams)
    bowling_team = st.selectbox("Bowling team", [team for team in teams if team != batting_team])
    venue = st.selectbox("Venue", venues)
    overs = st.slider("Completed overs", min_value=1, max_value=19, value=10)
    current_score = st.number_input("Current score", min_value=0, max_value=250, value=82, step=1)
    wickets = st.slider("Wickets fallen", min_value=0, max_value=9, value=2)
    st.markdown("<div class='section-kicker' style='margin-top:.8rem'>Recent momentum</div>", unsafe_allow_html=True)
    runs_last_5 = st.number_input("Runs in the last 5 overs", min_value=0, max_value=100, value=42, step=1)
    wickets_last_5 = st.slider("Wickets fallen in the last 5 overs", 0, 5, 1)

if HERO_IMAGE.exists():
    hero_image = base64.b64encode(HERO_IMAGE.read_bytes()).decode("ascii")
    hero_background = f"background-image:url('data:image/jpeg;base64,{hero_image}')"
else:
    hero_background = "background-image:linear-gradient(120deg,#132b3e,#07101c)"

st.markdown(
    f"""
    <section class="hero" style="{hero_background}" role="img" aria-label="Floodlit IPL cricket stadium">
      <div class="hero-copy">
        <div class="hero-tag"><span class="live-dot"></span> Match day analytics</div>
        <div class="eyebrow">Read the game. Project the total.</div>
        <h1>IPL first innings<br><span>score predictor</span></h1>
        <p>Turn the score, wickets and recent momentum into a live estimate of the innings finish.</p>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-kicker'>The matchup</div>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="matchup">
      <div class="team-card bat"><small>Batting first</small><strong>{html.escape(batting_team)}</strong></div>
      <div class="versus">VS</div>
      <div class="team-card bowl"><small>Bowling</small><strong>{html.escape(bowling_team)}</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_row = pd.DataFrame([{
    "venue": venue,
    "batting_team": batting_team,
    "bowling_team": bowling_team,
    "innings": 1,
    "overs_completed": overs,
    "current_score": current_score,
    "wickets_fallen": wickets,
    "runs_last_5": runs_last_5,
    "wickets_last_5": wickets_last_5,
    "current_run_rate": current_score / overs,
    "wickets_remaining": 10 - wickets,
}])[artifact["features"]]
prediction = max(float(current_score), float(artifact["pipeline"].predict(input_row)[0]))
prediction = int(round(prediction))

score_col, stats_col = st.columns([1.2, 1], gap="large")
with score_col:
    st.markdown(
        f"""
        <div class="score-panel">
          <div class="score-label">Projected first innings total</div>
          <div class="score-value">{prediction}</div>
          <div class="score-unit">runs</div>
          <div class="score-note">Estimated finish based on the game state so far</div>
          <span class="mini-stat">Current <b>{current_score}/{wickets}</b></span>
          <span class="mini-stat">Overs <b>{overs}.0</b></span>
          <span class="mini-stat">Run rate <b>{current_score / overs:.2f}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(current_score / max(prediction, 1), 1.0), text=f"{current_score} runs on the board")
    st.caption(f"About {max(0, prediction - current_score)} runs projected from here")

with stats_col:
    st.markdown("<div class='insight-title'>Model dugout</div>", unsafe_allow_html=True)
    metrics = artifact["metrics"]
    st.metric("Selected model", artifact["best_model"])
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Test MAE", f"{metrics['mae']:.1f} runs", help="Average absolute prediction error on held-out matches.")
    metric_b.metric("Test R²", f"{metrics['r2']:.3f}", help="Share of test-set score variation explained by the model.")

with st.expander("How this estimate is made"):
    st.write(
        "The scikit-learn pipeline combines the current score and run rate, wickets available, "
        "recent scoring, venue, and the teams. The test MAE is the average absolute miss on held-out "
        "matches. R² summarizes how much test-set score variation the model explains. This is an "
        "educational estimate, not a guarantee of a match outcome."
    )

st.markdown("<div class='insight-title' style='margin-top:1.25rem'>Inside the model</div>", unsafe_allow_html=True)
plot_a, plot_b = st.columns(2, gap="large")
for col, filename, caption in [
    (plot_a, "model_comparison.png", "MODEL COMPARISON · Lower MAE and higher R² are better"),
    (plot_b, "feature_importance.png", "MATCH DRIVERS · Held-out permutation importance"),
]:
    path = ROOT / "plots" / filename
    if path.exists():
        col.image(str(path), caption=caption, width="stretch")
    else:
        col.info("Run `python train_model.py` to generate training charts.")

st.markdown(
    f"<div class='footer-line'>Training rows: {artifact['training_rows']:,} &nbsp;·&nbsp; "
    f"Data: {html.escape(str(artifact['data_source']))} &nbsp;·&nbsp; "
    "Stadium photo: <a href='https://unsplash.com/photos/a-cricket-match-is-underway-at-a-stadium-CYbiE2T6Xtc' target='_blank' style='color:#aebbd0'>Unsplash · Zoshua Colah</a></div>",
    unsafe_allow_html=True,
)
