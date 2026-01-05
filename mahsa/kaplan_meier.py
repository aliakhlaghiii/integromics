"""
This module is designed to contain the scripts necessary for the following purposes:
1. Create stratified Kaplan-Meier plots based on specific features.
2. Create overall Kaplan-Meier plots for the entire cohort.
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