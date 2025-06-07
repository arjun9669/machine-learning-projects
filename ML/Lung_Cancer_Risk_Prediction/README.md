# 🩻 Lung Cancer Risk Prediction

This classification project predicts whether an individual is at high or low risk of developing lung cancer based on health indicators, symptoms, and lifestyle factors.  
A Random Forest classifier is used for high accuracy and explainability.

---

## 🎯 Objective

To detect and classify lung cancer risk early by analyzing:
- Smoking & alcohol consumption
- Pollution exposure
- Cough frequency, fatigue level, and chronic symptoms

---

## ❓ Problem Statement

Early-stage detection is critical in improving outcomes for lung cancer.  
This model helps flag high-risk individuals for further clinical evaluation.

---

## 📁 Files Included

| File Name                         | Description                                  |
|----------------------------------|----------------------------------------------|
| `lung_cancer_risk_prediction.ipynb` | Jupyter notebook with full pipeline         |
| `lung_cancer_risk_dataset.csv`      | Input dataset with risk indicators          |
| `lung_cancer_predictions.csv`       | (Optional) Output with predicted labels     |

---

## 📊 Visuals

- Risk class distribution (countplot)  
- Confusion matrix heatmap  
- Accuracy score & classification report (precision, recall, F1)

---

## 🧪 Techniques Used

- Train/test split (80/20)  
- RandomForestClassifier from sklearn  
- Accuracy, precision, recall, F1-score  
- Confusion Matrix using seaborn heatmap

---

## ▶️ How to Run

Install required packages:
```bash
pip install pandas matplotlib seaborn scikit-learn
