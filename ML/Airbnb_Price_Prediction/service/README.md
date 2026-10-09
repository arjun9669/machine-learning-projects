# Coastal Airbnb Price Predictor — Training Pipeline + FastAPI

A reproducible **educational** ML service built on the dataset already in this folder. The dataset is small and illustrative; it is **not representative of live Airbnb market prices**. No real-world performance or deployment claims are made.

## Features

- Train/holdout split (80/20) with a fixed random seed
- Missing-value handling and categorical encoding fitted **only on the training set**, preventing preprocessing leakage
- Random forest baseline with MAE, RMSE, R² and median-price baseline on held-out data
- Serialized sklearn pipeline and validated FastAPI inference endpoint
- Offline end-to-end test and GitHub Actions CI
- Docker image that trains the example artifact at build time

## Run locally

From `ML/Airbnb_Price_Prediction/service`:

```bash
python -m pip install -r requirements.txt
python train.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Then run:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"city":"Goa","bedrooms":2,"bathrooms":1,"distance_to_beach_km":2.5,"has_ac":true,"season":"Peak","rating":4.2}'
```

`GET /health` returns whether a trained model file is available. `POST /predict` returns an illustrative INR estimate, not a quote.

## Test and Docker

```bash
python -m unittest discover -s tests -v
# Run from the repository root (important: Dockerfile uses repo-relative COPY):
docker build -f ML/Airbnb_Price_Prediction/service/Dockerfile -t coastal-price-demo .
docker run --rm -p 8000:8000 coastal-price-demo
```

The generated `model.joblib` contains pickled Python objects. **Never load untrusted external model artifacts.** Rebuild from the provided CSV. The `metrics.json` file records measured holdout numbers after training, rather than making up scores in the README.

## Limitations

No authentication, rate limits, monitoring, drift detection, or external infrastructure deployment is included. This is an isolated local educational demo, not a production rental-pricing platform.
