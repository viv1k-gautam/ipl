"""Clean data, engineer features, compare sklearn regressors, and save artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ipl_data.csv"
MODEL_PATH = ROOT / "models" / "ipl_score_model.pkl"
PLOTS_PATH = ROOT / "plots"
TARGET = "final_score"
CATEGORICAL = ["venue", "batting_team", "bowling_team"]
NUMERIC = [
    "innings", "overs_completed", "current_score", "wickets_fallen",
    "runs_last_5", "wickets_last_5", "current_run_rate", "wickets_remaining",
]


def clean_and_engineer(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate inputs, remove invalid rows, and derive explainable cricket features."""
    required = CATEGORICAL + NUMERIC[:6] + [TARGET]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = frame.copy()
    for col in NUMERIC[:6] + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[TARGET, "overs_completed", "current_score", "wickets_fallen"])
    data = data[
        data["overs_completed"].between(1, 20)
        & (data["innings"] == 1)
        & data["current_score"].between(0, 300)
        & data["wickets_fallen"].between(0, 10)
        & data[TARGET].between(0, 350)
        & (data[TARGET] >= data["current_score"])
    ].copy()
    if len(data) < 30:
        raise ValueError(f"Need at least 30 valid innings rows; found {len(data)}.")
    data["current_run_rate"] = data["current_score"] / data["overs_completed"].clip(lower=1)
    data["wickets_remaining"] = 10 - data["wickets_fallen"]
    for col in CATEGORICAL:
        data[col] = data[col].fillna("Unknown").astype(str)
    X = data[CATEGORICAL + NUMERIC].replace([np.inf, -np.inf], np.nan)
    y = data[TARGET].astype(float)
    return X, y


def make_pipeline(model) -> Pipeline:
    """Keep imputation, encoding, scaling, and estimator together for safe inference."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ], verbose_feature_names_out=False)
    return Pipeline([("preprocess", preprocess), ("regressor", model)])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `python setup_data.py` first.")
    raw = pd.read_csv(DATA_PATH)
    X, y = clean_and_engineer(raw)
    if "match_id" in raw.columns:
        # Keep all snapshots from a real match together to avoid overly optimistic leakage.
        groups = raw.loc[X.index, "match_id"].astype(str)
        train_idx, test_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42
        ).split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    candidates = {
        "Ridge Regression": Ridge(alpha=8.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.9,
            random_state=42, n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.07, max_leaf_nodes=20,
            l2_regularization=1.0, random_state=42,
        ),
    }
    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results.append({
            "model": name,
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(root_mean_squared_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        })
        fitted[name] = pipeline
    results.sort(key=lambda row: (row["mae"], -row["r2"]))
    best_name = results[0]["model"]
    best_pipeline = fitted[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": best_pipeline,
        "features": CATEGORICAL + NUMERIC,
        "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC,
        "target": TARGET,
        "best_model": best_name,
        "metrics": results[0],
        "all_metrics": results,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "data_source": str(raw.get("data_source", pd.Series(["unspecified"])).mode().iloc[0]),
    }
    joblib.dump(artifact, MODEL_PATH)

    metrics = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#1d4ed8" if name == best_name else "#93c5fd" for name in metrics["model"]]
    axes[0].bar(metrics["model"], metrics["mae"], color=colors)
    axes[0].set_title("Mean Absolute Error (lower is better)")
    axes[0].set_ylabel("Runs")
    axes[1].bar(metrics["model"], metrics["r2"], color=colors)
    axes[1].set_title("R² score (higher is better)")
    axes[1].set_ylabel("R²")
    for ax in axes:
        ax.tick_params(axis="x", rotation=22)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("IPL first innings score model comparison")
    fig.tight_layout()
    fig.savefig(PLOTS_PATH / "model_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Permutation importance works with every candidate pipeline and is measured on held-out data.
    importance = permutation_importance(
        best_pipeline, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=8, random_state=42, n_jobs=-1,
    )
    imp = pd.Series(importance.importances_mean, index=X_test.columns).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    imp.plot.barh(ax=ax, color="#0f766e")
    ax.set_title(f"Top input features — {best_name}")
    ax.set_xlabel("Increase in MAE when feature is shuffled (runs)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_PATH / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (PLOTS_PATH / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Best model: {best_name}")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved charts to {PLOTS_PATH}")


if __name__ == "__main__":
    main()
