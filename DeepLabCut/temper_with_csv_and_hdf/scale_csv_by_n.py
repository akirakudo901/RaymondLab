# Author: Akira Kudo
# Created: 2024/03/15
# Last Updated: 2024/03/15

import os

import numpy as np
import pandas as pd

def rescale_csv_by_n(csv_path : str, outdir : str, n : int, new_scorer : str, 
                     also_do_hdf : bool=True, recalculate : bool=False):
    print("=============================")
    df = pd.read_csv(csv_path, index_col=[0,1,2], header=[0,1,2])
    df = df / 2
    filesafe_n = str(n).replace('.', 'point')
    old_scorer = os.path.basename(csv_path).replace('CollectedData_', '').replace('.csv', '')

    new_csvname = os.path.basename(csv_path).replace(
                                       old_scorer, new_scorer
                                           )#.replace(
                                       #'.csv', f'_rescaled_by_{filesafe_n}.csv'
                                       #    )
    rescaled_csv_path = os.path.join(outdir, new_csvname)
    if not os.path.exists(rescaled_csv_path) or recalculate:
        df.to_csv(rescaled_csv_path, index=True)
        # then rename the scorer & filename as index
        with open(rescaled_csv_path, 'r+') as f:
            lines = f.readlines()
            # rewrite the content of csv
            old_video_name = lines[3].split(',')[1]
            new_video_name = old_video_name + f'_rescaled_by_{filesafe_n}'
            lines[0] = lines[0].replace(old_scorer, new_scorer)
            lines = [l.replace(old_video_name, new_video_name) for l in lines]
            # save the result 
            f.seek(0); f.truncate() # delete what was already there
            f.writelines(lines)
        print(f'Csv generated at: {rescaled_csv_path}!')
    
    rescaled_hdf_path = rescaled_csv_path.replace('.csv', '.h5')
    if also_do_hdf and (not os.path.exists(rescaled_hdf_path) or recalculate):
        new_df = pd.read_csv(rescaled_csv_path, index_col=[0,1,2], header=[0,1,2])
        new_df.to_hdf(rescaled_hdf_path, key="rescaled", index=True)
        print(f"Hdf generated at: {rescaled_hdf_path}!")

if __name__ == "__main__":
    N = 0.5
    n_filesafe = str(N).replace('.', 'point')

    CSV_FILE = "CollectedData_Judy.csv"
    CSV_HOLDING_FOLDER_ROOT = "/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/labeled-data/full_dataset_including_belly"
#"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown Halfscale-Akira-2024-03-15/labeled-data"

    ABOVE_OUTDIR = "/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown Halfscale-Akira-2024-03-15/labeled-data"

    for d in os.listdir(CSV_HOLDING_FOLDER_ROOT):
        CSV_PATH = os.path.join(CSV_HOLDING_FOLDER_ROOT, d, CSV_FILE)
        outdir = os.path.join(ABOVE_OUTDIR, d + f"_rescaled_by_{n_filesafe}")
        print(f"Processing: {CSV_PATH}")
        try:
            rescale_csv_by_n(CSV_PATH, outdir=outdir,
                             n=N, recalculate=True, new_scorer="Akira")
        except Exception as e:
            print(e)
