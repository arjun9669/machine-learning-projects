# 🏘️ Airbnb Price Prediction

This educational regression project uses a small illustrative coastal-rentals dataset to model the relationship between listing features and nightly prices. It is not a validated live-market pricing tool.

### New reproducible API example

A separate [training + FastAPI service](./service/README.md) includes train/holdout metrics, preprocessing with a sklearn Pipeline, an inference endpoint, Docker instructions, and offline CI tests. It is a demonstration built on this repository's CSV, not an externally deployed service.  
It uses linear regression to model the relationship between key listing attributes and nightly price.

---

## 🎯 Objective

To accurately predict the price of an Airbnb listing based on:
- City and season
- Bedrooms and bathrooms
- Distance from beach
- AC availability and user ratings

---

## ❓ Problem Statement

Hosts on Airbnb often struggle with pricing strategy.  
This project helps them set competitive prices based on data-driven analysis.

---

## 📁 Files Included

| File Name                        | Description                                  |
|----------------------------------|----------------------------------------------|
| `airbnb_price_prediction.ipynb`  | Full notebook with model training & testing  |
| `airbnb_coastal_prices.csv`     | Input dataset with listing details           |
| `airbnb_price_predictions.csv`  | (Optional) Model-generated price output      |

---

## 📊 Visuals

- Correlation heatmap  
- Price distribution plot  
- Distance vs Price scatter  
- Regression line and R² score

---

## 🧪 Techniques Used

- One-Hot Encoding for categorical features  
- Linear Regression model (sklearn)  
- MAE, MSE, and R² Score for evaluation  
- Train/test split using sklearn

---

## ▶️ How to Run

Install required packages:
```bash
pip install pandas seaborn matplotlib scikit-learn
