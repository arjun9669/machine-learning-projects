# 🧬 Mall Customer App Usage Segmentation

This project performs clustering on mall customer behavior based on their app usage patterns and spending habits.  
It helps identify high-value customer groups to tailor personalized marketing strategies.

---

## 🎯 Objective

To segment mall customers into meaningful clusters using:
- App usage behavior
- Spending scores
- Visit patterns and demographics

---

## ❓ Problem Statement

Retail and shopping mall analytics need to understand:
- Which customers are frequent app users?
- Which segments have the highest spend potential?
- How can targeted offers be aligned with behavior?

Clustering helps answer these questions using unsupervised learning.

---

## 📁 Files Included

| File Name                         | Description                                 |
|----------------------------------|---------------------------------------------|
| `mall_customer_clustering.ipynb` | Full notebook with clustering workflow      |
| `mall_customer_app_usage.csv`    | Input dataset for clustering                |
| `mall_customer_clustered.csv`    | (Optional) Cluster-labeled export           |
| `kmeans_cluster_plot.png`        | Final cluster visualization (PCA-based)     |

---

## 📊 Visuals

- Elbow method chart to determine best `k`
- PCA-based cluster plot with color-coded groups
- Distribution plots for spending, visits, and age per cluster

---

## 🧪 Techniques Used

- 📐 **StandardScaler** for feature normalization  
- 📊 **KMeans Clustering**  
- 🎯 **PCA** for 2D visualization  
- 📈 **Cluster-wise analysis** using groupby & seaborn

---

## ▶️ How to Run

Install required packages:
```bash
pip install pandas matplotlib seaborn scikit-learn
