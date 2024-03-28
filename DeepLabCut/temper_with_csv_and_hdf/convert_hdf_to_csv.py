# Author: Akira Kudo

import os

import pandas as pd

def convert_hdf_to_csv(path_to_hdf, path_to_csv):
    df = pd.read_hdf(path_to_hdf)
    df.to_csv(path_to_csv, index=True)

if __name__ == "__main__":
    SAVE_DIR = r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown Halfscale-Akira-2024-03-15/labeled-data"
    HDF_FILENAME = "CollectedData_Judy.h5"
    
    data_folders = ["20220211070325_301533_f3_" + "_rescaled_by_0point5"]
                 #"20230102092905_363451_f1_openfield", 
                 #"20230107123308_362816_m1_openfield", 
                 #"20230107131118_363453_m1_openfield"]
    
    for data_hdf in data_folders:
        selected_filename_hdf = os.path.join(SAVE_DIR, data_hdf, HDF_FILENAME)
        selected_filename_csv = selected_filename_hdf.replace(".h5", ".csv")
    
        convert_hdf_to_csv(selected_filename_hdf, selected_filename_csv)
