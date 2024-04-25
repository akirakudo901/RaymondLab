# Author: Akira Kudo
# Created: 2024/04/24
# Last Updated: 2024/04/24

from typing import List

import numpy as np
import pandas as pd

from dlc_io.utils import read_dlc_csv_file

def remove_dlc_entry_by_probability_from_csv(
        dlc_path : str, threshold : float=0.8, bodyparts : List[str] = None
        ):
    """
    Reads a DLC dataframe from a given path, setting any entries
    of X and Y where likelihood is below threshold to np.nan.
    Only consider 'bodyparts' if given.

    :param str dlc_path: Path to dlc data holding csv.
    :param float threshold: Threshold for removing envries, defaults to 0.8.
    :param List[str] boryparts: The list of body parts for which we do the 
    removal. Defaults to None, every body part.
    """
    df = read_dlc_csv_file(dlc_path=dlc_path, include_scorer=False)
    return remove_dlc_entry_by_probability_from_dataframe(
        df, threshold=threshold, bodyparts=bodyparts)

def remove_dlc_entry_by_probability_from_dataframe(
        df : pd.DataFrame, threshold : float=0.8, bodyparts : List[str] = None
        ):
    """
    Reads a given DLC dataframe, setting any entries of X and Y 
    where likelihood is below threshold to np.nan.
    Only consider 'bodyparts' if given.

    :param pd.DataFrame df: Dataframe holding dlc data.
    :param float threshold: Threshold for removing envries, defaults to 0.8.
    :param List[str] boryparts: The list of body parts for which we do the 
    removal. Defaults to None, every body part.s
    """
    if type(threshold) != float: 
        raise Exception(f"threshold has to be a float but was {threshold} of type {type(threshold)}...")
    
    if bodyparts is None or len(bodyparts) == 0:
        bodyparts = np.unique(df.columns.get_level_values('bodyparts')).tolist()
    
    concerned_bodyparts = [bpt for bpt in bodyparts if bpt in bodyparts]
    
    for bpt in concerned_bodyparts:
        lklhd = df[(bpt, 'likelihood')]
        # filter based on threshold
        df[(bpt, 'x')][lklhd < threshold] = np.nan
        df[(bpt, 'y')][lklhd < threshold] = np.nan
    
    return df

if __name__ == "__main__":
    import os

    CSV_FOLDER = r"C:\Users\mashi\Desktop\temp\YAC\preBSOID_csv\snapshot-2010000\unfiltered"
    
    CSV_FILENAME = "20230113142714_392607_m1_openfieldDLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_2100000.csv"
    
    CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILENAME)

    filtered_df = remove_dlc_entry_by_probability_from_csv(dlc_path=CSV_PATH, threshold=0.9)
    print(filtered_df)