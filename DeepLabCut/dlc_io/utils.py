# Author: Akira Kudo
# Created: 2024/04/08
# Last Updated: 2024/04/08

import pandas as pd

def read_dlc_csv_file(dlc_path : str, include_scorer : bool=False):
    """
    Reads a dlc csv file object and returns a dataframe
    from it.

    :param str dlc_path: The path to dlc-data-holding csv.
    :param bool include_scorer: Whether to include scorer as index.
    Defaults to False.
    """
    header = [0,1,2] if include_scorer else [1,2]
    df = pd.read_csv(dlc_path, header=header, index_col=0)
    return df

if __name__ == "__main__":
    import os

    CSV_FOLDER = r"C:\Users\mashi\Desktop\RaymondLab\Experiments\B-SOID\Q175_Network\Q175_csv"
    CSV_FILENAME = r"20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"
    
    CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILENAME)

    df = read_dlc_csv_file(CSV_PATH)
    print(df.columns)
    df = read_dlc_csv_file(CSV_PATH, include_scorer=True)
    print(df.columns)