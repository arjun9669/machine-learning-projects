# 🏘️ Airbnb Price Prediction

This regression project aims to predict Airbnb rental prices in coastal cities using location, property features, and seasonality.  
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
