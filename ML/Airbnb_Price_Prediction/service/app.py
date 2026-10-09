"""Minimal FastAPI inference service for the illustrative price model."""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Coastal Rental Price Demo", version="1.0.0")


class ListingRequest(BaseModel):
    city: str = Field(min_length=1, max_length=80)
    bedrooms: int = Field(ge=1, le=20)
    bathrooms: int = Field(ge=1, le=20)
    distance_to_beach_km: float = Field(ge=0, le=1000, allow_inf_nan=False)
    has_ac: bool
    season: str = Field(min_length=1, max_length=40)
    rating: float = Field(ge=0, le=5, allow_inf_nan=False)


def model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", str(Path(__file__).with_name("model.joblib"))))


@lru_cache(maxsize=1)
def get_model(path: str):
    # joblib uses pickle internally: load ONLY the artifact we generated locally.
    return joblib.load(path)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": model_path().is_file()}


@app.post("/predict")
def predict(payload: ListingRequest):
    path = model_path()
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Model not trained. Run train.py first.")
    row = pd.DataFrame([{
        "City": payload.city,
        "Bedrooms": payload.bedrooms,
        "Bathrooms": payload.bathrooms,
        "Distance to Beach (km)": payload.distance_to_beach_km,
        "Has AC": int(payload.has_ac),
        "Season": payload.season,
        "Rating": payload.rating,
    }])
    try:
        artifact = get_model(str(path))
        predicted = float(artifact["pipeline"].predict(row[artifact["features"]])[0])
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Model unavailable for inference") from exc
    return {
        "predicted_price_inr": round(max(0.0, predicted), 2),
        "currency": "INR",
        "warning": "Educational estimate trained on an illustrative dataset; not a real rental quote.",
    }
