# Cox proportional hazards model and Kaplan-Meier Plots

In this folder, the following tasks are performed:

**preprocess.py**: 

- `calculate_delta_radiomics`: Functions for data extraction from Excel files, filtering by SUV threshold (2.5), and calculating temporal changes ($\Delta = \text{pre-LD} - \text{baseline}$).

**cox_analysis**:  

- `cox_univariate_analysis`: A standardized pipeline for univariate Cox Proportional Hazards regression, including feature scaling and proportional hazards assumption checking.

**kaplan_meier**:

- `plot_stratified_km`: A function to plot a stratified Kaplan-Meier plot based on the median of a specific feature
and marks the median survival point.

- `plot_overall_km`: A function to plot the overall Kaplan-Meier plot for the full cohort and marks the median.

**cox_kaplan_analysis.ipynb**: 

The primary research notebook documenting the end-to-end workflow, from data preprocessing (missing value imputation, variance filtering) to Kaplan-Meier visualization and statistical testing.

## Setup

### Requirements
- Python 3.x
- `numpy`, `pandas`, `pyyaml`, `matplotlib`, `lifelines`

Install dependencies (example):
```bash
pip install numpy pandas pyyaml matplotlib lifelines
```