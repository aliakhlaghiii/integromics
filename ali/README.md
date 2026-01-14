# Radiomics + Clinical Modeling (SVM / Random Forest)

This repository contains an end-to-end machine learning workflow to evaluate whether **delta radiomics** (Time B − Time A) adds predictive value beyond **clinical** variables and single-timepoint radiomics features. The analysis compares multiple feature configurations using **SVM (RBF)** and **Random Forest** models with **scikit-learn pipelines** and **GridSearchCV**.

> **Important limitation:** the current evaluation uses a **very small test set (n = 6; 3 per class)**, so all test-set metrics are highly unstable and should be interpreted cautiously.

---

## What’s inside

- **Notebook (main analysis)**
  - Full workflow from data loading → preprocessing/EDA → model training/tuning → evaluation → plots → interpretation.
- **Python modules**
  - `config.py`: loads `config.yaml` and validates required paths.
  - `data_io.py`: reads per-patient radiomics files (Time A/Time B), extracts the `suv2.5` segmentation row, computes delta features, and returns `A`, `B`, and `delta` DataFrames.
  - `models.py`: defines SVM/RF pipelines + a reusable GridSearchCV training/evaluation routine returning consistent result dictionaries.
  - `plots.py`: generates compact summary plots (heatmaps across metrics + confusion matrix strips) directly from the result dictionaries.

---

## Feature configurations evaluated

Each model is trained and evaluated separately for the following feature sets:

- `clinical` — clinical variables only  
- `clin+A` — clinical + radiomics at time A  
- `clin+B` — clinical + radiomics at time B  
- `clin+delta` — clinical + delta radiomics (B − A)

---

## Setup

### Requirements
- Python 3.x
- `numpy`, `pandas`, `pyyaml`, `matplotlib`
- `scikit-learn`

Install dependencies (example):
```bash
pip install numpy pandas pyyaml matplotlib scikit-learn
```

## Author
Ali Akhlaghi  

Email:
a.akhlaghi@st.hanze.nl

