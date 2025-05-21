
# 📊 economic_profile_clustering

This project explores and analyzes countries' economic and demographic indicators through **unsupervised learning**. Using **DBSCAN (Density-Based Spatial Clustering)**, countries are grouped based on similarities in features like income, health expenditure, child mortality, and more. The results are visualized using **PCA (Principal Component Analysis)** for better interpretation.

---

## 🌍 Objective

To identify meaningful economic clusters among countries based on key socio-economic metrics — without using predefined labels or classes.

---

## 📁 Dataset

- `Country-data.csv` — Contains data on:
  - Income per capita
  - Health expenditure (% of GDP)
  - Life expectancy
  - Child mortality rate
  - Inflation
  - Imports / Exports
  - Fertility rate
  - GDP per capita

---

## 🛠️ Methods Used

- **StandardScaler** – for feature normalization
- **DBSCAN** – for clustering countries based on density and distance
- **PCA** – for reducing high-dimensional data into 2D for visualization
- **Seaborn / Matplotlib** – for plotting clustered data

---

## 📌 Key Highlights

- No need to define the number of clusters (unlike K-means)
- Detection of **outlier countries** that don't fit into any group (`cluster = -1`)
- Cluster profiles showing average income, health levels, mortality rates, etc.
- Supports deeper economic insight for data-driven analysis

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/economic_profile_clustering.git
   cd economic_profile_clustering
