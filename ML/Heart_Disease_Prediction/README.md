# ❤️ Heart Disease Prediction

> A classification machine learning project that predicts whether a patient is likely to have heart disease based on medical and demographic features.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Model](https://img.shields.io/badge/Model-XGBoost%20%7C%20LogReg%20%7C%20DT-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

## 🧠 Problem Statement

Heart disease is one of the leading causes of death globally. Early diagnosis is critical for effective treatment.  
This project uses classification models to predict the presence of heart disease based on patient features like age, cholesterol level, ECG results, and exercise-induced angina.

---

## 📂 Files

- `Heart_Disease_Prediction.ipynb` — Full notebook with data prep, training, evaluation & ROC  
- `heart_disease_prediction.csv` — Dataset with 500 synthetic patient records  
- `README.md` — Documentation

---

## 🧪 Features Used

- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Max Heart Rate  
- Exercise-Induced Angina  
- ST Depression (Oldpeak)  
- ST Slope  

---

## ✅ Tasks Performed

- Label encoding for categorical variables  
- Train-test split (80/20)  
- Trained three models: Logistic Regression, Decision Tree, XGBoost  
- Evaluated with Accuracy, Confusion Matrix, Classification Report  
- Plotted ROC Curve for XGBoost model

---

## 📈 Model Results

- ✅ **Best Accuracy:** ~87% with XGBoost  
- 📉 ROC-AUC: ~0.91  
- 📊 Visual comparison of all models

---

## 🚀 How to Run

```bash
pip install xgboost
jupyter notebook Heart_Disease_Prediction.ipynb
