# Author: Akira Kudo

import os

import pandas as pd

from DeepLabCut.temper_with_csv_and_hdf.convert_csv_hdf import convert_csv_to_hdf
from select_n_images_based_on_label import select_n_images_based_on_label

DATA_DIR = r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/labeled-data/full_dataset_including_belly"
SAVE_DIR = r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/labeled-data"
CSV_FILENAME = "CollectedData_Judy.csv"


UNNAMED_ONE, UNNAMED_TWO = "Unnamed: 1", "Unnamed: 2"

def save_to_csv(df, filepath):
    df.to_csv(filepath, index=False)
    # open the same csv and rearrange the first row
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # insert a header that creates a multi-index when read again into csv
        new_header = lines[0].replace(UNNAMED_ONE, '').replace(UNNAMED_TWO, '')
        new_header_entries = new_header.split(',')
        scorer = new_header_entries[3]
        new_header_array = new_header_entries[:4] + [scorer] * (len(new_header_entries) - 4) + ['\n']
        new_header = ','.join(new_header_array)
    # reopen to overwrite
    with open(filepath, 'w') as f:
        # write the same file with new header inserted to the same path
        f.writelines([new_header] + lines[1:])

def save_to_hdf(hdf_filepath, csv_filepath):
    convert_csv_to_hdf(csv_filepath, hdf_filepath)

data_csvs = ["20220211070325_301533_f3_", 
             "20230102092905_363451_f1_openfield", 
             "20230107123308_362816_m1_openfield", 
             "20230107131118_363453_m1_openfield"]
coord_paths = [os.path.join(DATA_DIR, filename, CSV_FILENAME) for filename in data_csvs]
label_paths = [os.path.join(DATA_DIR, filename, filename + "_storage.csv") for filename in data_csvs]

for data_csv, coord_csv, label_csv in zip(data_csvs, coord_paths, label_paths):
    curr_df_array = []
    #Label. 0 : Wall rear, 1 : Grooming, 2 : Other
    for label, num_sample in zip([0, 1, 2], [13, 13, 999]):
        selected_df = select_n_images_based_on_label(coord_csv, label_csv, label=label, n=num_sample, sample_seed=5)
        header = selected_df[0:2]
        selected_df = selected_df[2:]
        curr_df_array.append(selected_df)
    curr_df = pd.concat([header] + curr_df_array)
    # selected_filename_csv = os.path.join(SAVE_DIR, data_csv, CSV_FILENAME)
    selected_filename_csv = os.path.join(SAVE_DIR, data_csv, CSV_FILENAME.replace(".csv", "_selected.csv"))
    selected_filename_h5 = selected_filename_csv.replace(".csv", ".h5")
    
    save_to_csv(curr_df, selected_filename_csv)
    save_to_hdf(selected_filename_h5, selected_filename_csv)
