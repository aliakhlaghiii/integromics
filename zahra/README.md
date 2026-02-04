# Radiomics and Clinical Feature Integration for Survival Prediction in CAR T-cell Therapy

## Overview

This repository contains a complete machine learning workflow designed to study the role of PET-based radiomics features in short-term survival prediction for patients with diffuse large B-cell lymphoma (DLBCL) receiving CAR T-cell therapy. In addition to standard clinical variables, the pipeline examines whether changes in radiomics features over time (delta radiomics between Time A and Time B) provide useful additional information.

Due to the small sample size (n = 30) and the very limited independent test set (n = 6), all analyses are exploratory and intended only for hypothesis generation. The results should not be interpreted as clinically actionable. The focus of this work is to systematically compare single-timepoint radiomics with longitudinal radiomics features within a controlled experimental framework.

## Data
- Retrospective cohort: **30 DLBCL patients**
- Train/Test split: **24 / 6**

## Feature configurations:
  - **X0:** Clinical variables only
  - **X1:** Clinical + baseline radiomics
  - **X2:** Clinical + pre-lymphodepletion radiomics
  - **X3:** Clinical + delta radiomics (B − A) 

## Models
- **Logistic Regression**
  - Penalties: None, L1, L2, Elastic Net
  - Optional L1-based feature selection

- **K-Nearest Neighbors (KNN)**

## Evaluation metrics:
- Accuracy
- ROC-AUC
- F1-score

## Repository Structure

- `preprocessor.py` – data loading and preprocessing
- `logistic_regression.py` – Logistic Regression pipeline
- `knn_classification.py` – KNN analysis
- `analysis_ML_z.ipynb` – main analysis notebook (figures and interpretation)
- `visualization.py` – plotting utilities
- `*.csv` – exported result summaries

## Setup
### Requirements
- Python 3.x
- numpy
- pandas
- pyyaml
- matplotlib
- scikit-learn

### Installation
```bash
pip install numpy pandas pyyaml matplotlib scikit-learn
```

## Usage
1. Configure parameters in `config.yaml`
2. Run preprocessing:
   ```bash
   python preprocessor.py
   ```
3. Run models:
   ```bash
   python logistic_regression.py
   python knn_classification.py
   ```

## Notes
- Model performance is highly sensitive to train–test splitting due to the very small test set.
- Apparent performance gains from radiomics features should be interpreted with caution.
- This codebase is intended for research and methodological exploration only and is **not** suitable for clinical decision-making.

## Author
Zahra Taheri Hanjanai
Hanze University of Applied Sciences  
Email: z.taheri.hanjani@st.hanze.nl
