# 💳 Loan Approval Prediction using XGBoost

> Classification project using XGBoost to predict whether a customer’s loan should be approved based on personal and financial attributes.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🧠 Problem Statement

Banks and financial institutions want to automate the loan approval process. This project uses XGBoost to classify whether a loan application should be approved based on features like income, credit score, dependents, and education level.

---

## 📁 Project Structure

- `Loan_Approval_XGBoost.ipynb` — Full notebook with model training and evaluation  
- `loan_approval_xgb.csv` — Dataset (500 rows, generated for demo)  
- `README.md` — This documentation file

---

## 🔍 Features Used

- Age  
- Income  
- Loan Amount  
- Credit Score  
- Dependents  
- Education (Graduate / Not Graduate)  
- Self_Employed (Yes / No)

---

## ✅ Tasks Performed

- Data Cleaning and Label Encoding  
- Train-test split  
- XGBoost Classifier model training  
- Evaluation using Accuracy, Classification Report, Confusion Matrix  
- ROC-AUC Curve Plot

---

## 📊 Model Performance

- ✅ Accuracy: ~85%  
- 📉 ROC-AUC: ~0.90  
- 📈 Visualized results using ROC Curve and Confusion Matrix

---

## 🚀 How to Run

```bash
pip install xgboost
jupyter notebook Loan_Approval_XGBoost.ipynb
