# Author: Akira Kudo
# Created: 2024/03/15
# Last Updated: 2024/04/20

import os
import sys

import pandas as pd

def scale_dlc_label_csv_by_n(csv_path : str, n : float, outdir : str, 
                             new_video_name : str=None, new_name : str=None, 
                             new_scorer : str=None, also_do_hdf : bool=False,
                             overwrite : bool=False):
    """
    Scales the numerical values contained in a given dlc label csv by n, outputting it
    to outdir while setting it a new name (generated if not given).
    Also can reanem the 'scorer' entries if required, as well as output the 
    resulting csv into hdf.

    :param str csv_path: Path to csv holding dlc data.
    :param float n: Scaling factor.
    :param str outdir: Path to target directory for output.
    :param str new_video_name: Name of videos the label data is indicated to come from,
    defaults to no change.
    :param str new_name: New name of generated file, defaults to auto-generation.
    :param str new_scorer: A new 'scorer' for the scorer entry, defaults to no change.
    :param bool also_do_hdf: Whether to also output hdf of the same name. Defaults to False.
    :param bool overwrite: Overwrite when target output file exists, defaults to False.
    """
    df = pd.read_csv(csv_path, index_col=[0,1,2], header=[0,1,2])
    df = df * n
    # get old_scoere if new_scorer is given
    if new_scorer is not None:
        old_scorer = os.path.basename(csv_path).replace('CollectedData_', '').replace('.csv', '')
    # if new_name is not given, define it
    if new_name is None:
        filesafe_n = str(n).replace('.', 'point')

        new_name = os.path.basename(csv_path).replace(
                                        old_scorer, new_scorer
                                            ).replace(
                                        '.csv', f'_rescaled_by_{filesafe_n}.csv'
                                           )
    rescaled_csv_path = os.path.join(outdir, new_name)
    
    if not overwrite and os.path.exists(rescaled_csv_path):
        print(f"Target file already exists... : {rescaled_csv_path}")
        sys.exit(1)
    
    df.to_csv(rescaled_csv_path, index=True)
    # then rename the scorer & filename as index
    with open(rescaled_csv_path, 'r+') as f:
        lines = f.readlines()
        # rewrite the content of csv
        # deal with the scorer 
        if new_scorer is not None:
            lines[0] = lines[0].replace(old_scorer, new_scorer)
        # as well as the video name
        if new_video_name is not None:        
            old_video_name = lines[3].split(',')[1]
            lines = [l.replace(old_video_name, new_video_name) for l in lines]
        # save the result 
        f.seek(0); f.truncate() # delete what was already there
        f.writelines(lines)
    print(f'Csv generated at: {rescaled_csv_path}!')
    
    rescaled_hdf_path = rescaled_csv_path.replace('.csv', '.h5')
    if also_do_hdf:
        if not overwrite and os.path.exists(rescaled_hdf_path):
            print(f"Target file already exists... : {rescaled_hdf_path}")
            sys.exit(1)

        new_df = pd.read_csv(rescaled_csv_path, index_col=[0,1,2], header=[0,1,2])
        new_df.to_hdf(rescaled_hdf_path, key="rescaled", index=True)
        print(f"Hdf generated at: {rescaled_hdf_path}!")


if __name__ == "__main__":
    N = 0.5
    n_filesafe = str(N).replace('.', 'point')

    CSV_FILE = "CollectedData_Judy.csv"
    CSV_HOLDING_FOLDER_ROOT = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\video_related\data"

    OUTDIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\video_related\data\result"

    CSV_PATH = os.path.join(CSV_HOLDING_FOLDER_ROOT, CSV_FILE)

    scale_dlc_label_csv_by_n(csv_path=CSV_PATH, n=N, 
                             outdir=OUTDIR,
                             new_video_name='new_video', 
                             new_name="CollectedData_Akira.csv", 
                             new_scorer='Akira', 
                             also_do_hdf=True)