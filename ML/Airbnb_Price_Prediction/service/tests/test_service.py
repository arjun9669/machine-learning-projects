"""Offline end-to-end smoke test: train then predict through the HTTP interface."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))

from train import train_model
from app import app, get_model

DATA = SERVICE.parent / "airbnb_coastal_prices.csv"
EXAMPLE = {
    "city": "Goa",
    "bedrooms": 2,
    "bathrooms": 1,
    "distance_to_beach_km": 2.5,
    "has_ac": True,
    "season": "Peak",
    "rating": 4.2,
}


class ServiceTests(unittest.TestCase):
    def test_full_train_and_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.joblib"
            metrics = Path(tmp) / "metrics.json"
            result = train_model(DATA, model, metrics)
            self.assertTrue(model.exists())
            self.assertTrue(metrics.exists())
            self.assertGreater(result["holdout_rows"], 0)
            self.assertGreaterEqual(result["holdout_mae_inr"], 0)
            with patch.dict(os.environ, {"MODEL_PATH": str(model)}):
                get_model.cache_clear()
                with TestClient(app) as client:
                    res = client.post("/predict", json=EXAMPLE)
                    self.assertEqual(res.status_code, 200, res.text)
                    self.assertEqual(res.json()["currency"], "INR")
                    self.assertGreaterEqual(res.json()["predicted_price_inr"], 0)
                    bad = client.post("/predict", json={**EXAMPLE, "rating": 8.0})
                    self.assertEqual(bad.status_code, 422)
                get_model.cache_clear()

    def test_model_missing_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"MODEL_PATH": str(Path(tmp) / "not-there.joblib")}):
                with TestClient(app) as client:
                    self.assertEqual(client.get("/health").json()["model_available"], False)
                    self.assertEqual(client.post("/predict", json=EXAMPLE).status_code, 503)


if __name__ == "__main__":
    unittest.main()
