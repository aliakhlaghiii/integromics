"""
Preprocessing module for radiomics and clinical data analysis
"""

import os
import pandas as pd
import numpy as np


def calculate_delta_radiomics(data_folder_path):
    """
    Reads radiomics data from subfolders (Time A and Time B), 
    filters for 'suv2.5' segmentation, calculates the delta (B - A) for numeric features.

    Args:
        data_folder_path (str): The path to the main folder containing patient subfolders.

    Returns:
        tuple: (delta_radiomics_results, a_radiomics, b_radiomics)
            - delta_radiomics_results: Delta radiomics (B - A) as DataFrame
            - a_radiomics: Radiomics at time A as DataFrame
            - b_radiomics: Radiomics at time B as DataFrame
    """
    all_delta_radiomics = {}
    A_radiomics, B_radiomics = {}, {}

    # 1. Iterate through all items in the main data folder
    for patient_folder_name in os.listdir(data_folder_path):
        patient_path = os.path.join(data_folder_path, patient_folder_name)
        
        # Ensure it is actually a directory (a patient folder)
        if os.path.isdir(patient_path):
            print(f"--- Processing {patient_folder_name} ---")
            
            # Initialize paths for Time A and Time B files
            file_A_path = None
            file_B_path = None
            
            # 2. Find the radiomics files for Time A and Time B in the patient folder
            for filename in os.listdir(patient_path):
                path_excel = os.path.join(patient_path, filename)

                # Assuming filenames contain '_A' or '_B' (case-insensitive) + .xlsx
                upper_name = path_excel.upper()
                if '_A' in upper_name and path_excel.endswith('.xlsx'):
                    file_A_path = path_excel
                elif '_B' in upper_name and path_excel.endswith('.xlsx'):
                    file_B_path = path_excel

            if file_A_path and file_B_path:
                try:
                    # 3. Read and preprocess the data
                    df_A = pd.read_excel(file_A_path)
                    df_B = pd.read_excel(file_B_path)
                    
                    # 4. Filter for the 'suv2.5' segmentation row, take columns from 23 onwards
                    row_A = df_A[df_A['Segmentation'].str.contains('suv2.5')].iloc[0, 23:]
                    row_B = df_B[df_B['Segmentation'].str.contains('suv2.5')].iloc[0, 23:]

                    # 5. Convert to numeric, coercing errors to NaN
                    numeric_A = pd.to_numeric(row_A, errors='coerce')
                    numeric_B = pd.to_numeric(row_B, errors='coerce')

                    # 6. Calculate Delta Radiomics (Time B - Time A)
                    delta_radiomics = numeric_B - numeric_A
                    
                    # Store as dicts, dropping NaNs
                    all_delta_radiomics[patient_folder_name] = delta_radiomics.dropna().to_dict()
                    A_radiomics[patient_folder_name] = numeric_A.dropna().to_dict()
                    B_radiomics[patient_folder_name] = numeric_B.dropna().to_dict()

                    print(f"Successfully calculated radiomics and delta radiomics for {patient_folder_name}.")

                except Exception as e:
                    print(f"Error processing files for {patient_folder_name}: {e}")
            else:
                print(f"Could not find both A and B files in {patient_folder_name}.")

    # Convert dicts to DataFrames (patients = rows, features = columns)
    A_df = pd.DataFrame.from_dict(A_radiomics, orient='index')
    B_df = pd.DataFrame.from_dict(B_radiomics, orient='index')
    delta_df = pd.DataFrame.from_dict(all_delta_radiomics, orient='index')

    return delta_df, A_df, B_df


def clean_and_suffix_radiomics(delta_radiomics_results, a_radiomics, b_radiomics):
    """
    Clean and prepare dataframes by dropping columns with any NaN values,
    resetting index, and adding suffixes.

    Args:
        delta_radiomics_results: Delta radiomics DataFrame
        a_radiomics: Time A radiomics DataFrame
        b_radiomics: Time B radiomics DataFrame

    Returns:
        tuple: (delta_radiomics_results, a_radiomics, b_radiomics) - all cleaned and with suffixes
    """
    # Clean and prepare dataframes
    # by dropping columns with any NaN values and resetting index
    # to keep only the complete cases (some patients have 99 columns with NaNs, but 43 are always present)
    # we'll work with those 43.
    for df in [delta_radiomics_results, a_radiomics, b_radiomics]:
        df.dropna(axis=1, how='any', inplace=True)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
        df['id'] = df['id'].astype(int)

    a_radiomics = a_radiomics.add_suffix('_a')
    b_radiomics = b_radiomics.add_suffix('_b')
    
    return delta_radiomics_results, a_radiomics, b_radiomics


def prepare_and_clean_clinical_data(clinic_data, delta_radiomics_results):
    """
    Prepare clinical data by creating cleaned IDs,
    filtering patients, dropping unnecessary columns, and handling disease diagnosis.

    Args:
        clinic_data: Raw clinical DataFrame
        delta_radiomics_results: Delta radiomics DataFrame (with 'id' column)

    Returns:
        pd.DataFrame: Cleaned clinical data
    """
    # Make a copy to avoid modifying original
    clinic_data = clinic_data.copy()
    
    # creating a cleaned id column so we can merge later with radiomics data
    clinic_data['id_cleaned'] = [value[-3:] for value in clinic_data['record_id'].values]

    # type conversion to int for merging with radiomics data
    patient_ids = clinic_data['id_cleaned'].values[1:].astype(int)
    
    # find patients that are in both datasets
    # values starts from 1 to skip the comment row
    intercept = [id for id in delta_radiomics_results['id'] if id in patient_ids]
    clinic_data['id_cleaned'] = ['ID'] + patient_ids.tolist()
    clinic_data_cleaned = clinic_data[clinic_data['id_cleaned'].isin(intercept)]
    clinic_data_cleaned = clinic_data_cleaned.copy()  # Make a proper copy
    clinic_data_cleaned.reset_index(drop=True, inplace=True)
    
    # dropping columns with all NaN values
    clinic_data_cleaned = clinic_data_cleaned.dropna(axis=1, how='all')
    
    # we don't need factor columns for modelling as they are encoded already
    factors = [factor for factor in clinic_data_cleaned.columns if 'factor' in factor]
    comments = [comm for comm in clinic_data_cleaned.columns if 'comment' in comm]
    locations = [loc for loc in clinic_data_cleaned.columns if 'loc' in loc]
    
    # these are highly correlated features with bmi
    correlated = ['scr_height', 'scr_weight']
    indicators = ['indication_ldh_uln','indication_age_60','indication_extran_sites', 'indication_extran_invol']

    # cause of death columns are not needed
    cause_of_death = [cause for cause in clinic_data_cleaned.columns if '_cause' in cause]
    
    # One-hot encode disease diagnosis if column exists
    if 'indication_dis_diagnosis.factor' in clinic_data_cleaned.columns:
        disease = pd.get_dummies(clinic_data_cleaned['indication_dis_diagnosis.factor']).astype(int)
    else:
        disease = pd.DataFrame()

    # Only drop columns that actually exist
    drop_columns = cause_of_death + factors + ['record_id','scr_date_tb1stmeeting', 'indication_dis_diagnosis'] + comments + correlated + indicators
    drop_columns = [col for col in drop_columns if col in clinic_data_cleaned.columns]
    clinic_data_cleaned.drop(columns=drop_columns, inplace=True)

    # Add disease encoding if it exists
    if not disease.empty:
        clinic_data_cleaned = pd.concat([clinic_data_cleaned, disease], axis=1)
    
    # replacing 'NE' with NaN to create valid missing values
    clinic_data_cleaned.replace({'NE': np.nan}, inplace=True)

    nans = clinic_data_cleaned.isna().sum().sort_values(ascending=False)

    # droping columns with more than half nans
    drop_nans = nans[nans >= clinic_data_cleaned.shape[0]/2].index
    clinic_data_cleaned = clinic_data_cleaned.drop(columns=drop_nans)
    
    return clinic_data_cleaned


def convert_clinical_data_types(clinic_data_cleaned):
    """
    Convert data types in clinical data.

    Args:
        clinic_data_cleaned: Clinical DataFrame

    Returns:
        pd.DataFrame: Clinical data with correct types
    """
    # Assuming clinic_data_filtered is the DataFrame you want to convert
    date_columns = [date for date in clinic_data_cleaned.columns if ('date' in date) or ('start' in date) or ('stop' in date)]
    
    # 1. Use convert_dtypes() for general automatic inference
    # This function automatically converts to best possible dtypes (e.g., object to string, int64 to Int64, float64 to Float64)
    # It's particularly useful for handling missing values using pandas' nullable dtypes (e.g., pd.NA).
    print("Applying general type conversion...")

    # 2. Force remaining object columns that look like numbers to numeric
    for col in clinic_data_cleaned.columns:
        if col not in date_columns:
            # Attempt to convert to numeric.
            # this is to fix a typo in columns where , is used instead of .
            if clinic_data_cleaned[col].dtype == 'object':
                clinic_data_cleaned[col] = pd.to_numeric(clinic_data_cleaned[col].str.replace(',','.'), errors='raise')
        else: 
            clinic_data_cleaned[col] = pd.to_datetime(clinic_data_cleaned[col], errors='coerce')
        
    print("\nAutomatic type conversion complete.")
    
    return clinic_data_cleaned


def remove_zero_variance_features(clinic_data_cleaned):
    """
    Remove zero variance columns.

    Args:
        clinic_data_cleaned: Clinical DataFrame

    Returns:
        pd.DataFrame: Clinical data without zero variance features
    """
    variances = clinic_data_cleaned.select_dtypes(include=np.number).var().sort_values()

    # zero variance columns are not useful for modelling so I am dropping them
    zero_var = variances[variances == 0].index
    clinic_data_cleaned = clinic_data_cleaned.drop(columns=zero_var)
    
    return clinic_data_cleaned


def impute_clinical_missing_values(clinic_data_cleaned):
    """
    Impute missing values with the median for numeric columns.

    Args:
        clinic_data_cleaned: Clinical DataFrame

    Returns:
        pd.DataFrame: Clinical data with imputed values
    """
    # Impute missing values with the median for numeric columns
    for col in clinic_data_cleaned.select_dtypes(include=np.number).columns:
        median_value = clinic_data_cleaned[col].median()
        clinic_data_cleaned[col].fillna(median_value, inplace=True)
    
    return clinic_data_cleaned


def drop_clinical_date_columns(clinic_data_cleaned):
    """
    Drop date-related columns.

    Args:
        clinic_data_cleaned: Clinical DataFrame

    Returns:
        pd.DataFrame: Clinical data without date columns
    """
    # there are date related column that still have nans, but we will not use them for ml modelling
    date_columns = ['ae_summ_icans_stop_v2','ae_summ_icans_start_v2', 
                    'surv_death_date',
                    'surv_date',
                    'surv_time_bestresponse_car', 
                    'surv_prog_after_car',
                    'surv_prog_date',
                    'ae_summ_crs_start_v2', 'ae_summ_crs_stop_v2',
                    'tr_car_ld_start',
                    'cli_st_lab_date']
    clinic_data_cleaned.drop(columns=date_columns, inplace=True)

    # we also don't need the remaining date columns for modelling
    date_columns = [col for col in clinic_data_cleaned.columns if 'date' in col]
    clinic_data_cleaned.drop(columns=date_columns, inplace=True)
    
    return clinic_data_cleaned



def create_all_modeling_datasets(clinic_data_cleaned, delta_radiomics_results, a_radiomics, b_radiomics):
    """
    Create all modeling datasets (X0, X1, X2, X3) and target variable y.

    Args:
        clinic_data_cleaned: Cleaned clinical DataFrame
        delta_radiomics_results: Delta radiomics DataFrame
        a_radiomics: Time A radiomics DataFrame (with '_a' suffix)
        b_radiomics: Time B radiomics DataFrame (with '_b' suffix)

    Returns:
        dict: Dictionary with keys 'X0', 'X1', 'X2', 'X3', 'y'
    """
    # all the clinical and delta radiomics and single point radiomics data combined
    modelling_data = pd.concat([clinic_data_cleaned, delta_radiomics_results, a_radiomics, b_radiomics], axis=1)

    # we need to drop the last row, as the patient's clinical data is not available
    modelling_data = modelling_data.iloc[:-1,:]

    only_clinic = clinic_data_cleaned.drop(columns=['id_cleaned','surv_status'])

    no_delta_radiomics = pd.concat([clinic_data_cleaned, a_radiomics, b_radiomics], axis=1)
    # to eliminate patient 95 who there's no clinical data for
    no_delta_radiomics = no_delta_radiomics.iloc[:-1,:]
    
    X_with_delta = modelling_data.drop(columns=['id_a','id_b','id_cleaned','id','surv_status']) 
    X_without_delta = no_delta_radiomics.drop(columns=['id_a','id_b','id_cleaned','surv_status'])

    X_with_a_radiomics = pd.concat([only_clinic, a_radiomics], axis=1).iloc[:-1,:]
    X_with_b_radiomics = pd.concat([only_clinic, b_radiomics], axis=1).iloc[:-1,:]

    target_variable = "surv_status"
    y = modelling_data[target_variable]

    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)

    y = modelling_data["surv_status"]

    X0_clinical_only  = only_clinic.copy()
    X1_clinical_A     = X_with_a_radiomics.copy()
    X2_clinical_B     = X_with_b_radiomics.copy()
    X3_clinical_delta = X_with_delta.copy()

    datasets = {
        "X0_clinical_only": X0_clinical_only,
        "X1_clinical_A": X1_clinical_A,
        "X2_clinical_B": X2_clinical_B,
        "X3_clinical_delta": X3_clinical_delta,
        "y": y
    }
    
    return datasets


def full_preprocessing_pipeline(DATA_DIR, clinic_data):
    """
    Complete preprocessing pipeline that runs ALL steps.

    Args:
        DATA_DIR: Path to radiomics data folder
        clinic_data: Raw clinical DataFrame

    Returns:
        dict: Dictionary containing X0, X1, X2, X3, and y
    """
    print("\n" + "="*70)
    print("STARTING FULL PREPROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Calculate delta radiomics
    print("\n[Step 1/10] Calculating delta radiomics...")
    delta_radiomics_results, a_radiomics, b_radiomics = calculate_delta_radiomics(DATA_DIR)
    
    print("\n--- Final Results Summary ---")
    for patient, delta_data in list(delta_radiomics_results.items())[:3]:
        print(f"\n{patient} Delta Radiomics ({len(delta_data)} features):")
        print(dict(list(delta_data.items())[:5]))
    
    # Step 2: Clean and add suffixes to radiomics
    print("\n[Step 2/10] Cleaning radiomics dataframes and adding suffixes...")
    delta_radiomics_results, a_radiomics, b_radiomics = clean_and_suffix_radiomics(
        delta_radiomics_results, a_radiomics, b_radiomics
    )
    
    # Step 3: Prepare and clean clinical data
    print("\n[Step 3/10] Preparing and cleaning clinical data...")
    clinic_data_cleaned = prepare_and_clean_clinical_data(clinic_data, delta_radiomics_results)
    
    # Step 4: Convert data types
    print("\n[Step 4/10] Converting clinical data types...")
    clinic_data_cleaned = convert_clinical_data_types(clinic_data_cleaned)
    
    # Step 5: Remove zero variance features
    print("\n[Step 5/10] Removing zero variance features...")
    clinic_data_cleaned = remove_zero_variance_features(clinic_data_cleaned)
    
    # Step 6: Impute missing values
    print("\n[Step 6/10] Imputing missing values...")
    clinic_data_cleaned = impute_clinical_missing_values(clinic_data_cleaned)
    
    # Step 7: Drop date columns
    print("\n[Step 7/10] Dropping date columns...")
    clinic_data_cleaned = drop_clinical_date_columns(clinic_data_cleaned)
    
    # Step 8: Check no NaNs remain
    print(f"\nConfirming no NaNs remain: {clinic_data_cleaned.isna().sum().sum()}")
    
    # Step 9: Create all modeling datasets
    print("\n[Step 9/10] Creating all modeling datasets...")
    datasets = create_all_modeling_datasets(
        clinic_data_cleaned, delta_radiomics_results, a_radiomics, b_radiomics
    )
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"X0 (Clinical only): {datasets['X0_clinical_only'].shape}")
    print(f"X1 (Clinical + A): {datasets['X1_clinical_A'].shape}")
    print(f"X2 (Clinical + B): {datasets['X2_clinical_B'].shape}")
    print(f"X3 (Clinical + Delta): {datasets['X3_clinical_delta'].shape}")
    print(f"Target (y): {datasets['y'].shape}")
    
    return datasets

