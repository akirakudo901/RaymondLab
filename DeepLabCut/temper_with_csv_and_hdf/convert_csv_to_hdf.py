# Author: Akira Kudo

import os

import pandas as pd

def convert_csv_to_hdf(path_to_csv, path_to_hdf):
    df = pd.read_csv(path_to_csv, index_col=[0,1,2], header=[0,1,2])
    df.to_hdf(path_to_hdf, key="selected", index=False)

if __name__ == "__main__":
    SAVE_DIR = r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/labeled-data"
    CSV_FILENAME = "CollectedData_Judy.csv"
    
    data_csvs = ["20220211070325_301533_f3_", 
                 "20230102092905_363451_f1_openfield", 
                 "20230107123308_362816_m1_openfield", 
                 "20230107131118_363453_m1_openfield"]
    
    for data_csv in data_csvs:
        selected_filename_csv = os.path.join(SAVE_DIR, data_csv, CSV_FILENAME)
        selected_filename_hdf = selected_filename_csv.replace(".csv", ".h5")
    
        convert_csv_to_hdf(selected_filename_csv, selected_filename_hdf)
