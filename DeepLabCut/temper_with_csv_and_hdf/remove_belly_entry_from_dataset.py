# Author: Akira Kudo
# Functions for removing the "belly" label from datasets where the belly is already saved.

import os

import numpy as np
import pandas as pd

def remove_belly_entry_from_dataframe(df_path : str, save_dir : str, save_result : bool=True):
    """
    Removes the "belly" entries from an already labeled dataframe.
    Deals with both csv and h5 by looking at the extension.
    """
    H5_FILETYPE, CSV_FILETYPE = "h5", "csv"
    filetype = H5_FILETYPE if df_path.endswith(".h5") else (CSV_FILETYPE if df_path.endswith(".csv") else None)
    if filetype is None: raise Exception("Passed file has to have extension h5 or csv; try again!")
    
    if filetype == H5_FILETYPE:
        df = pd.read_hdf(df_path)
    elif filetype == CSV_FILETYPE:
        df = pd.read_csv(df_path, index_col=0, header=[0,1,2])
    bpt_row = df.columns.get_level_values("bodyparts")
    non_belly_columns = np.where(bpt_row != "belly")[0]
    df = df.iloc[:, non_belly_columns]
    
    if save_result:
        new_filename = os.path.basename(df_path).replace(f".{filetype}", f"_bellyRemoved.{filetype}")
        full_savepath = os.path.join(save_dir, new_filename)
        print(f"Saving {new_filename} under {save_dir}...")
        if filetype == H5_FILETYPE:
            df.to_hdf(full_savepath, key="belly_removed_data", mode="w")
        elif filetype == CSV_FILETYPE:
            df.to_csv(full_savepath)
        print("Successful!")

if __name__ == "__main__":
    ROOT = "/media/Data/Raymond Lab/Python_Scripts"
    DATA_DIR = "data"
    H5_PATH  = os.path.join(ROOT, DATA_DIR, "CollectedData_Judy.h5")
    CSV_PATH = os.path.join(ROOT, DATA_DIR, "CollectedData_Judy.csv")
    SAVE_DIR = os.path.join(ROOT, DATA_DIR)
    remove_belly_entry_from_dataframe(H5_PATH,  save_dir=SAVE_DIR, save_result=True)
    remove_belly_entry_from_dataframe(CSV_PATH, save_dir=SAVE_DIR, save_result=True)
