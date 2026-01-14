"""
This module is designed to calculate delta radiomics features from radiomics data at two different time points.

"""
import os
import pandas as pd

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






