# 🔍 Model Comparison with GridSearchCV

This project compares three machine learning classifiers using hyperparameter tuning with GridSearchCV:
- Decision Tree
- Random Forest
- XGBoost

Each model is evaluated based on accuracy and classification metrics, with an ROC curve plotted for XGBoost.

---

## 📁 Dataset
- **File**: `loan_approval_xgb.csv`
- **Target Variable**: `Loan_Status`
- **Features**: Income, Employment, Education, Credit History, etc.

---

## ⚙️ Models & Hyperparameters

| Model         | Parameters Tuned          |
|---------------|---------------------------|
| Decision Tree | `max_depth: [3, 5, 7]`    |
| Random Forest | `n_estimators`, `max_depth` |
| XGBoost       | `n_estimators`, `max_depth` |

---

## 📊 Evaluation
- Accuracy & classification report for each model
- ROC Curve + AUC for XGBoost

---

## ✅ Libraries Used
`scikit-learn`, `xgboost`, `matplotlib`, `pandas`, `numpy`
