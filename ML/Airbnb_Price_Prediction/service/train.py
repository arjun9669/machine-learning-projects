"""Train a reproducible educational Airbnb price model (no target leakage)."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Price (INR)"
NUMERIC = ["Bedrooms", "Bathrooms", "Distance to Beach (km)", "Has AC", "Rating"]
CATEGORICAL = ["City", "Season"]
FEATURES = NUMERIC + CATEGORICAL


def train_model(data: Path, output: Path, metrics: Path) -> dict:
    frame = pd.read_csv(data)
    missing = set(FEATURES + [TARGET]) - set(frame)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    frame = frame.dropna(subset=[TARGET])
    if len(frame) < 30:
        raise ValueError("At least 30 labeled rows are required for this demo")
    X, y = frame[FEATURES], frame[TARGET]
    if not np.isfinite(y).all():
        raise ValueError("Target contains nonfinite values")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocess", preprocessing),
        ("regressor", RandomForestRegressor(
            n_estimators=80, min_samples_leaf=2, random_state=42, n_jobs=1,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    result = {
        "training_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "holdout_mae_inr": round(float(mean_absolute_error(y_test, predictions)), 2),
        "holdout_rmse_inr": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 2),
        "holdout_r2": round(float(r2_score(y_test, predictions)), 4),
        "baseline_median_mae_inr": round(float(mean_absolute_error(
            y_test, np.full(len(y_test), y_train.median())
        )), 2),
        "random_state": 42,
        "caveat": "Illustrative dataset only; no real-world pricing validity claimed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": FEATURES}, output)
    metrics.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "airbnb_coastal_prices.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "model.joblib")
    parser.add_argument("--metrics", type=Path, default=Path(__file__).resolve().parent / "metrics.json")
    args = parser.parse_args()
    print(json.dumps(train_model(args.data, args.output, args.metrics), indent=2))
