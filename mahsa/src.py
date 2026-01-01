"""
This module is designed to contain the scripts necessary for the following purposes:
1. Calculate delta radiomics features from radiomics data at two different time points.
2. Perform univariate Cox regression analysis on the radiomic features.
3. Create stratified Kaplan-Meier plots based on specific features.
4. Create overall Kaplan-Meier plots for the entire cohort.
"""
import os
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times
import numpy as np


def calculate_delta_radiomics(data_folder_path):
    """
    Reads radiomics data from subfolders (Time A and Time B), filters for 'suv2.5' 
    segmentation, calculates the delta (B - A) for numeric features, and stores
    the results in a dictionary per patient.

    Args:
        data_folder_path (str): The path to the main folder containing patient subfolders.

    Returns:
        A tuple of three pd.DataFrames:
            - all_delta_radiomics: DataFrame with delta radiomics features (B - A)
            - A_radiomics: DataFrame with Time A radiomics features
            - B_radiomics: DataFrame with Time B radiomics features
        
    """
    all_delta_radiomics = {}
    A_radiomics, B_radiomics = {}, {}

    # 1. Iterate through all items in the main data folder
    for patient_folder_name in os.listdir(data_folder_path):
        patient_path = os.path.join(data_folder_path, patient_folder_name)
        
        # Ensure it is actually a directory (a patient folder)
        if os.path.isdir(patient_path):            
            # Initialize paths for Time A and Time B files
            file_A_path = None
            file_B_path = None
            
            # 2. Find the radiomics files for Time A and Time B in the patient folder
            for filename in os.listdir(patient_path):
                path_excel = os.path.join(patient_path, filename)
                # NOTE: Assuming the files are named consistently and contain 'A' or 'B' 
                # to identify the time point. Adjust this logic if needed.
  
                if '_A' in path_excel.upper() and path_excel.endswith('.xlsx'):
                        file_A_path = path_excel
                elif '_B' in path_excel.upper() and path_excel.endswith('.xlsx'):
                        file_B_path = path_excel
            if file_A_path and file_B_path:
                try:
                    # 3. Read and preprocess the data
                    
                    # Read Excel files and transpose them (assuming features are in columns 
                    # and metadata/values in rows; pandas reads the first row as header)
                    # We assume 'segmentation' is one of the columns after reading.
                    df_A = pd.read_excel(file_A_path)
                    df_B = pd.read_excel(file_B_path)
                    
                    # 4. Filter for the 'suv2.5' segmentation row
                    # NOTE: the column containing 'suv2.5' is named 'Segmentation'
                    # and the feature names are in the other columns.
                    # filtering the columns fro 23 onwards to get only feature values
                    row_A = df_A[df_A['Segmentation'].str.contains('suv2.5')].iloc[0, 23:]
                    row_B = df_B[df_B['Segmentation'].str.contains('suv2.5')].iloc[0, 23:]

                    # Create a Series of only the numeric feature values for A and B
                    
                    # Convert to numeric, coercing errors to NaN (just in case)
                    numeric_A = pd.to_numeric(row_A, errors='coerce')
                    numeric_B = pd.to_numeric(row_B, errors='coerce')

                    # 6. Calculate Delta Radiomics (Time B - Time A)
                    delta_radiomics = numeric_B - numeric_A
                    
                    
                    # Convert the resulting pandas Series into a standard Python dictionary
                    # and store it under the patient's ID
                    # dropna() to remove any features that resulted in NaN
                    all_delta_radiomics[patient_folder_name] = delta_radiomics.dropna().to_dict()
                    A_radiomics[patient_folder_name] = numeric_A.dropna().to_dict()
                    B_radiomics[patient_folder_name] = numeric_B.dropna().to_dict()

                except Exception as e:
                    print(f"Error processing files for {patient_folder_name}: {e}")
            else:
                print(f"Could not find both A and B files in {patient_folder_name}.")
    A_radiomics = pd.DataFrame.from_dict(A_radiomics, orient='index')
    B_radiomics = pd.DataFrame.from_dict(B_radiomics, orient='index')
    all_delta_radiomics = pd.DataFrame.from_dict(all_delta_radiomics, orient='index')

    print("Delta radiomics calculation completed.")

    return all_delta_radiomics, A_radiomics, B_radiomics



def cox_univariate_analysis(X, T_col='T', E_col='E', var_threshold=1.0):
    """
    Perform univariate Cox regression on radiomic features with variance filtering.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input dataframe with radiomic features + T, E columns
    T_col : str, default='T'
        Duration column name
    E_col : str, default='E' 
        Event column name
    var_threshold : float, default=1.0
        Minimum variance threshold for feature selection
    
    Returns:
    --------
    pd.DataFrame
        Results with feature, HR, p-value, CI
    """
    # Select radiomic features (exclude T, E)
    radiomic_features = [col for col in X.columns if col not in [T_col, E_col]]
    
    # Variance filtering
    ftr_var = X[radiomic_features].var()
    selected_features = ftr_var[ftr_var > var_threshold].index.tolist()
    
    results = []
    X_scaled = X.copy()
    for feature in selected_features:
        try:
            # Scale feature
            scaler = StandardScaler()
            cph = CoxPHFitter()
            # ravel() to convert back to 1D array
            X_scaled[feature] = scaler.fit_transform(X[[feature]]).ravel()
            
            # Fit Cox model
            cph.fit(X_scaled[[feature, T_col, E_col]], 
                   duration_col=T_col, event_col=E_col)
            
            summary = cph.summary
            
            # test proportional hazards assumption only for significant features
            if summary.loc[feature, 'p'] <= 0.05:
                cph.check_assumptions(X_scaled[[feature, T_col, E_col]], p_value_threshold=0.05, show_plots=False)
                

            results.append({
                'feature': feature,
                'HR': summary.loc[feature, 'exp(coef)'],
                'p_value': summary.loc[feature, 'p'],
                'lower_CI': summary.loc[feature, 'exp(coef) lower 95%'],
                'upper_CI': summary.loc[feature, 'exp(coef) upper 95%'],
                'z_score': summary.loc[feature, 'z']
            })
            
        except Exception as e:
            print(f"Could not fit Cox model for {feature}: {e}")

    
    return pd.DataFrame(results)



def plot_stratified_km(df, duration_col, event_col, feature_name, title_suffix=""):
    """
    Creates a stratified Kaplan-Meier plot based on the median of a specific feature
    and marks the median survival point.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing duration, event, and feature columns.
    duration_col : str
        Name of the column representing duration until event or censoring.
    event_col : str
        Name of the column representing event occurrence (1 if event occurred, 0 if censored
    feature_name : str
        Name of the feature column to stratify by (median split).
    title_suffix : str, optional
        Suffix to add to the plot title for context.

    Returns:
    --------
    None
    """
    # 1. Define Groups based on Median Split
    cutoff = df[feature_name].median()
    masks = {
        f'High {feature_name} (>{cutoff:.2f})': df[feature_name] > cutoff,
        f'Low {feature_name} (≤{cutoff:.2f})': df[feature_name] <= cutoff
    }
    
    # 2. Setup Plot
    plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()

# Define color mapping for consistency
    color_map = {
        'High': 'tab:orange', # High feature value group
        'Low': 'tab:blue'     # Low feature value group
    }

    # 3. Iterate through groups to fit, plot, and mark medians
    for label, mask in masks.items():
        T_sub = df.loc[mask, duration_col]
        E_sub = df.loc[mask, event_col]
        
        if len(T_sub) == 0:
            continue

        # Determine color based on whether "High" or "Low" is in the label
        current_color = color_map['High'] if 'High' in label else color_map['Low']
        
        kmf.fit(T_sub, event_observed=E_sub, label=label)
        
        # Pass the explicit color to the plot function
        kmf.plot_survival_function(ax=ax, color=current_color)
        
        # Calculate median statistics
        median_time = kmf.median_survival_time_
        ci_df = median_survival_times(kmf.confidence_interval_)
        
        # Plot the median point and helper lines using the explicit color
        if np.isfinite(median_time):
            # Horizontal line to 50%
            plt.plot([0, median_time], [0.5, 0.5], color=current_color, 
                     linestyle='--', linewidth=1, alpha=0.6)
            # Vertical line down to X-axis
            plt.plot([median_time, median_time], [0, 0.5], color=current_color, 
                     linestyle='--', linewidth=1, alpha=0.6)
            # The Median Point (Point determining the median)
            plt.scatter([median_time], [0.5], color=current_color, s=80, 
                        zorder=5, edgecolors='white',
                        label=f'Median {label}: {median_time:.0f}d')

        # Print statistics to console for Results section
        lower_ci = ci_df.iloc[0, 0]
        upper_ci = ci_df.iloc[0, 1]
        print(f"\n{label} Median {title_suffix}: {median_time} days")
        print(f"95% CI: {lower_ci} - {upper_ci} days")
    
    # 4. Log-Rank Test
    group_keys = list(masks.keys())
    results = logrank_test(
        df.loc[masks[group_keys[0]], duration_col], 
        df.loc[masks[group_keys[1]], duration_col], 
        event_observed_A=df.loc[masks[group_keys[0]], event_col], 
        event_observed_B=df.loc[masks[group_keys[1]], event_col]
    )
    
    # 5. Formatting
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim([0, 1.05]) # Ensure y-axis starts at 0
    plt.axhline(0.5, color='black', lw=0.5, alpha=0.3) # 50% reference line
    plt.title(f"KM Stratified by {feature_name} ({title_suffix})\nLog-Rank p-value: {results.p_value:.4f}")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='best', fontsize='small')
    plt.show()
    




def plot_overall_km(df, duration_col, event_col, title_suffix=""):
    """
    Creates an overall Kaplan-Meier plot for the full cohort and marks the median.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing duration and event columns.
    duration_col : str
        Name of the column representing duration until event or censoring.
    event_col : str
        Name of the column representing event occurrence (1 if event occurred, 0 if censored
    title_suffix : str, optional
        Suffix to add to the plot title for context.

    Returns:
    --------
    KaplanMeierFitter
        The fitted KaplanMeierFitter object for further analysis if needed.
    """
    # 1. Setup Plot
    plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    
    # 2. Fit and Plot
    kmf.fit(df[duration_col], event_observed=df[event_col], label=f'Overall {title_suffix}')
    line = kmf.plot_survival_function(ax=ax, color='teal') # Using a neutral color for overall
    color = 'teal'
    
    # 3. Calculate Median Statistics
    median_time = kmf.median_survival_time_
    ci_df = median_survival_times(kmf.confidence_interval_)
    
    # 4. Visualizing the Median Point
    if np.isfinite(median_time):
        # Horizontal line to 50%
        plt.plot([0, median_time], [0.5, 0.5], color=color, linestyle='--', linewidth=1, alpha=0.6)
        # Vertical line down to X-axis
        plt.plot([median_time, median_time], [0, 0.5], color=color, linestyle='--', linewidth=1, alpha=0.6)
        # The Median Point
        plt.scatter([median_time], [0.5], color=color, s=60, zorder=5, 
                    label=f'Median: {median_time} days')
    
    # 5. Formatting for the report
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim([0, 1.05])
    plt.axhline(0.5, color='black', lw=0.5, alpha=0.3) # 50% reference line
    plt.title(f"Overall {title_suffix} (Full Cohort, n={len(df)})")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(loc='best')
    
    # 6. Printing Numerical Results for the Results Section
    lower_ci = ci_df.iloc[0, 0]
    upper_ci = ci_df.iloc[0, 1]
    print(f"--- {title_suffix} Summary ---")
    print(f"Median Survival: {median_time} days")
    print(f"95% CI: {lower_ci} - {upper_ci} days\n")
    
    plt.show()
    
    return kmf