# Project: Prognostic Value of Baseline and Pre-Lymphodepletion PET/CT Imaging in DLBCL Patients Undergoing CAR T-Cell Therapy

## About the repo

This repository contains several complementary analysis pipelines to investigate whether PET-based radiomics features add predictive value beyond clinical variables for short‑term survival prediction in patients with diffuse large B‑cell lymphoma (DLBCL) treated with CAR T‑cell therapy at the UMCG center. The work is exploratory due to the very small cohort (n = 30) and is intended for hypothesis generation, not clinical decision‑making. The code covers survival analysis with Cox models and Kaplan–Meier plots, classical machine‑learning classifiers (Logistic Regression, KNN), and models using SVM and Random Forest with scikit‑learn pipelines.

**Key components:**

- **Folder Mahsa: Cox Proportional Hazards & Kaplan-Meier analysis** (Cox univariate regression, stratified KM plots, survival visualization).  

- **Folders Ali and Zahra: Machine learning classification** (SVM RBF, Random Forest, Logistic Regression, KNN for survival prediction).  

- **Shared in all analysis folders: Radiomics feature extraction** (baseline, pre-lymphodepletion, and delta radiomics Time pre-LD (B) - Time baseline (A)).  

- **Shared in all analysis folders: Clinical data integration** (clinical variables + radiomics for predictive modeling). 

- **Folder models_performance:** Plots of ML model results according to 3 metrics (F1-score, Accuracy and ROC-AUC)


## Installing

1. **Clone the repository:** 

```bash
   git clone https://github.com/Mahsa-Zf/integromics.git
```

2. **Create virtual environment:**

```bash
# Recommended: conda environment
conda create -n radiomics-analysis python=3.13
conda activate radiomics-analysis

# Or venv
python -m venv radiomics-env
source radiomics-env/bin/activate  # Linux/macOS
# radiomics-env\Scripts\activate   # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Data access:**
    - Restricted dataset available to qualified researchers upon approval from the institutional data access committee.


5. **How to run the analyses:**
    - By running the notebooks in each folder, you'd be able to replicate the results.

## Authors

- **Ali Akhlaghi** (a.akhlaghi@st.hanze.nl)
- **Mahsa Zamanifard** (m.zamani.fard@st.hanze.nl)
- **Zahra Taheri Hanjanai** (z.taheri.hanjan@st.hanze.nl)

## Acknowledgements

We would like to thank **Kylie Keijzer, PhD candidate**, University Medical Center Groning-en for her supervision and valuable scientific and clinical guidance.


## License

This project is licensed under the **MIT License**


