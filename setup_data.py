"""Create a clean innings-level IPL training dataset.

The script first tries a public, innings-level CSV URL configured through
IPL_DATA_URL. If it cannot be downloaded, it creates reproducible synthetic
innings snapshots with plausible IPL scoring patterns. Synthetic data is
clearly marked in the output so it is never mistaken for real match records.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "ipl_data.csv"
IPL_OUTPUT = ROOT / "data" / "ipl.csv"
MATCHES_URL = "https://raw.githubusercontent.com/waquararshadkhan/ipl-analytics/main/data/matches.csv"
DELIVERIES_URL = "https://raw.githubusercontent.com/waquararshadkhan/ipl-analytics/main/data/deliveries.csv"
MATCHES_PATH = ROOT / "data" / "matches.csv"
DELIVERIES_PATH = ROOT / "data" / "deliveries.csv"
RANDOM_SEED = 42
FEATURES = [
    "venue", "batting_team", "bowling_team", "innings", "overs_completed",
    "current_score", "wickets_fallen", "runs_last_5", "wickets_last_5",
]
TARGET = "final_score"


def generate_synthetic_data(n_innings: int = 5000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate realistic innings snapshots at several stages of an innings."""
    rng = np.random.default_rng(seed)
    teams = [
        "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bengaluru",
        "Kolkata Knight Riders", "Rajasthan Royals", "Sunrisers Hyderabad",
        "Delhi Capitals", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants",
    ]
    venues = [
        "Wankhede Stadium", "M. A. Chidambaram Stadium", "Eden Gardens",
        "M. Chinnaswamy Stadium", "Narendra Modi Stadium", "Arun Jaitley Stadium",
        "Rajiv Gandhi International Stadium", "Sawai Mansingh Stadium",
        "Punjab Cricket Association Stadium", "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    ]
    venue_bonus = rng.normal(0, 0.35, len(venues))
    rows: list[dict[str, object]] = []
    for _ in range(n_innings):
        batting_idx, bowling_idx = rng.choice(len(teams), size=2, replace=False)
        innings = 1
        venue_idx = int(rng.integers(len(venues)))
        # Sample checkpoints throughout an innings; the model predicts its eventual total.
        overs = int(rng.choice(np.arange(5, 20)))
        wickets = int(rng.integers(0, min(8, overs // 2 + 1)))
        base_rate = float(np.clip(rng.normal(8.35 + venue_bonus[venue_idx], 0.65), 6.4, 10.1))
        # A modest second-innings chase effect represents more conservative/volatile play.
        rate = base_rate + (0.12 if innings == 2 else 0.0) + rng.normal(0, 0.38)
        score = int(np.clip(overs * rate - wickets * rng.uniform(0.4, 1.2) + rng.normal(0, 6), 20, 190))
        recent_overs = min(5, overs)
        runs_last_5 = int(np.clip(recent_overs * (rate + rng.normal(0, 1.0)) - rng.uniform(0, 7), 8, 75))
        wickets_last_5 = int(rng.integers(0, min(4, wickets) + 1))
        # Remaining scoring depends on wickets, current rate, venue, and natural innings variance.
        remaining = 20 - overs
        acceleration = rng.normal(0.55, 0.27) if wickets <= 4 else rng.normal(0.1, 0.3)
        future_rate = np.clip(rate + acceleration + venue_bonus[venue_idx] * 0.2, 5.0, 12.0)
        expected_remaining = remaining * future_rate
        expected_remaining -= max(0, wickets - 5) * remaining * 0.28
        final_score = int(np.clip(round(score + expected_remaining + rng.normal(0, 10 + remaining * 0.32)), score, 280))
        rows.append({
            "venue": venues[venue_idx], "batting_team": teams[batting_idx],
            "bowling_team": teams[bowling_idx], "innings": innings,
            "overs_completed": overs, "current_score": score,
            "wickets_fallen": wickets, "runs_last_5": runs_last_5,
            "wickets_last_5": wickets_last_5, TARGET: final_score,
            "data_source": "synthetic_fallback",
        })
    return pd.DataFrame(rows)


def load_real_data(url: str) -> pd.DataFrame:
    """Load an innings-level CSV and validate the required input and target columns."""
    frame = pd.read_csv(url)
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Downloaded CSV is not innings-level; missing columns: {sorted(missing)}")
    frame = frame[FEATURES + [TARGET]].copy()
    frame = frame[frame["innings"] == 1].copy()
    if frame.empty:
        raise ValueError("Downloaded CSV contains no first-innings rows.")
    frame["data_source"] = "downloaded"
    return frame


def download_kaggle_source() -> pd.DataFrame:
    """Fetch the public mirror of the Kaggle IPL Complete Dataset (2008–2024)."""
    MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MATCHES_PATH.exists():
        urlretrieve(MATCHES_URL, MATCHES_PATH)
    if not DELIVERIES_PATH.exists():
        urlretrieve(DELIVERIES_URL, DELIVERIES_PATH)

    matches = pd.read_csv(MATCHES_PATH, usecols=["id", "venue"])
    balls = pd.read_csv(DELIVERIES_PATH)
    needed = {"match_id", "inning", "batting_team", "bowling_team", "over", "total_runs", "is_wicket"}
    missing = needed - set(balls.columns)
    if missing:
        raise ValueError(f"Kaggle deliveries file is missing columns: {sorted(missing)}")
    balls = balls[balls["inning"] == 1].copy()
    balls = balls.merge(matches, left_on="match_id", right_on="id", how="inner")
    balls["over_runs"] = pd.to_numeric(balls["total_runs"], errors="coerce").fillna(0)
    balls["is_wicket"] = pd.to_numeric(balls["is_wicket"], errors="coerce").fillna(0)
    # Aggregate each over first; rolling windows then represent the last five completed overs.
    overs = balls.groupby(
        ["match_id", "over"], as_index=False, sort=True
    ).agg(
        over_runs=("over_runs", "sum"),
        over_wickets=("is_wicket", "sum"),
        venue=("venue", "first"),
        batting_team=("batting_team", "first"),
        bowling_team=("bowling_team", "first"),
    )
    overs = overs.sort_values(["match_id", "over"]).reset_index(drop=True)
    by_match = overs.groupby("match_id", sort=False)
    overs["current_score"] = by_match["over_runs"].cumsum()
    overs["wickets_fallen"] = by_match["over_wickets"].cumsum()
    overs["runs_last_5"] = overs.groupby("match_id", sort=False)["over_runs"].transform(
        lambda values: values.rolling(5, min_periods=1).sum()
    )
    overs["wickets_last_5"] = overs.groupby("match_id", sort=False)["over_wickets"].transform(
        lambda values: values.rolling(5, min_periods=1).sum()
    )
    totals = overs.groupby("match_id")["over_runs"].sum()
    overs["final_score"] = overs["match_id"].map(totals)
    overs["innings"] = 1
    overs["overs_completed"] = overs["over"] + 1
    # Old franchise names are mapped to the current names displayed by the app.
    overs = overs.replace({
        "batting_team": {
            "Delhi Daredevils": "Delhi Capitals",
            "Kings XI Punjab": "Punjab Kings",
            "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        },
        "bowling_team": {
            "Delhi Daredevils": "Delhi Capitals",
            "Kings XI Punjab": "Punjab Kings",
            "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        },
    })
    columns = FEATURES + [TARGET]
    snapshots = overs.loc[overs["overs_completed"].between(1, 19), columns].copy()
    snapshots["match_id"] = overs.loc[snapshots.index, "match_id"]
    snapshots["data_source"] = "Kaggle IPL Complete Dataset 2008-2024"
    if snapshots.empty:
        raise ValueError("Kaggle IPL source produced no first-innings checkpoints.")
    return snapshots


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("IPL_DATA_URL", "").strip()
    if url:
        try:
            frame = load_real_data(url)
            print(f"Loaded {len(frame):,} innings records from IPL_DATA_URL.")
        except Exception as exc:  # Network and schema failures both use a safe fallback.
            print(f"Could not use IPL_DATA_URL ({exc}); generating synthetic fallback data.")
            frame = generate_synthetic_data()
    else:
        try:
            frame = download_kaggle_source()
            print(f"Built {len(frame):,} first-innings snapshots from Kaggle IPL data.")
        except Exception as exc:
            print(f"Kaggle dataset download/processing failed ({exc}); generating synthetic fallback data.")
            frame = generate_synthetic_data()
    frame.to_csv(OUTPUT, index=False)
    frame.to_csv(IPL_OUTPUT, index=False)
    print(f"Wrote {len(frame):,} records to {OUTPUT} and {IPL_OUTPUT}")


if __name__ == "__main__":
    main()
