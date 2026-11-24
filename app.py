"""Streamlit interface for the saved IPL innings score regression pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ipl_score_model.pkl"
st.set_page_config(page_title="IPL First Innings Score Predictor", page_icon="🏏", layout="wide")


@st.cache_resource
def load_artifact():
    """Load once per Streamlit process; the preprocessing is bundled in the pipeline."""
    return joblib.load(MODEL_PATH)


st.title("🏏 IPL First Innings Score Predictor")
st.caption("Estimate a first innings total from the match situation so far.")

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
    st.header("Match situation")
    batting_team = st.selectbox("Batting team", teams)
    bowling_team = st.selectbox("Bowling team", [team for team in teams if team != batting_team])
    venue = st.selectbox("Venue", venues)
    overs = st.slider("Completed overs", min_value=1, max_value=19, value=10)
    current_score = st.number_input("Current score", min_value=0, max_value=250, value=82, step=1)
    wickets = st.slider("Wickets fallen", min_value=0, max_value=9, value=2)
    runs_last_5 = st.number_input("Runs in the last 5 overs", min_value=0, max_value=100, value=42, step=1)
    wickets_last_5 = st.slider("Wickets fallen in the last 5 overs", 0, 5, 1)

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

left, right = st.columns([1, 1])
with left:
    st.subheader("Predicted first innings total")
    st.metric("Projected score", f"{prediction} runs", help="The prediction is constrained to be at least the current score.")
    st.progress(min(current_score / max(prediction, 1), 1.0), text=f"{current_score} runs scored so far")
    st.caption(f"Approximate runs remaining: {max(0, prediction - current_score)}")
with right:
    st.subheader("Model snapshot")
    metrics = artifact["metrics"]
    st.metric("Selected model", artifact["best_model"])
    st.metric("Test MAE", f"{metrics['mae']:.1f} runs")
    st.metric("Test R²", f"{metrics['r2']:.3f}")

with st.expander("How to read the estimate"):
    st.write(
        "The estimate combines the score and run rate so far, wickets available, recent scoring, "
        "venue, and team identities. Test MAE is the average absolute miss on a held-out test "
        "split; R² summarizes how much test-set score variation the model explains. The result "
        "is an educational estimate, not a guarantee of a match outcome."
    )

st.subheader("Training insights")
plot_a, plot_b = st.columns(2)
for col, filename, caption in [
    (plot_a, "model_comparison.png", "Model comparison (lower MAE and higher R² are better)"),
    (plot_b, "feature_importance.png", "Held-out permutation importance for the selected model"),
]:
    path = ROOT / "plots" / filename
    if path.exists():
        col.image(str(path), caption=caption, width="stretch")
    else:
        col.info("Run `python train_model.py` to generate training charts.")

st.caption(f"Training rows: {artifact['training_rows']:,} · Data source: {artifact['data_source']}")
