# Author: Akira Kudo
# Created: 2024/04/27
# Last Updated: 2024/04/27

import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlc_io.utils import read_dlc_csv_file
from utils_to_be_replaced_oneday import get_mousename

"""
This script attempts to validate newly generated DLC csvs in batch by:
 1) Checking the duration of the file, identifying up to what 5 minute time range
    we can use for total distance traveled.
 2) Identifying portions in the csv with low probability, to find when mouse is for example
    removed from cage before video ends.
 3) Rendering mouse trajectory plots to ensure there is no otherwise noticeable problematic
    noise.
 4) Maybe showing some stats of the result?
"""

TRUNCATED_SUFFIX_NO_EXT = '_trunc'
TRUNCATE_THRESHOLD = 0.8

def check_csv_duration(csv_path : str, interval : int=300, fps : int=40):
    """
    Checks the file duration of the given DLC csv, identyfing how many complete 
    'interval' second blocks are contained.
    Also returns at what time the csv ends.

    :param str csv_path: Path to csv holding DLC data.
    :param int interval: Interval in seconds we chunk data to see if it is complete, 
    defaults to 300 (5 min).
    :param int fps: Frame-per-second of video passed to DLC, defaults to 40.

    :return int num_blocks: Max number of blocks completed given interval & fps.
    :return int endtime: The number of frames at which the csv ends.
    """
    df = read_dlc_csv_file(csv_path, include_scorer=False)
    num_timepoints = df.shape[0]
    num_blocks = num_timepoints // (interval * fps)
    endtime = num_timepoints
    return num_blocks, endtime

def visualize_low_probability_on_all_bodyparts(csv_path : str,  
                                               save_dir : str,
                                               save_name : str,
                                               title : str,
                                               threshold : float=TRUNCATE_THRESHOLD,
                                               save_figure : bool=True,
                                               show_figure : bool=True):
    """
    Visualizes where probabilities less than 'threshold' happens on all body parts.
    
    :return np.ndarray: Frame indices where body parts likelihoods were all < threshold.
    """
    df = read_dlc_csv_file(csv_path, include_scorer=False)
    # find all columns with entries pertaining to 'likelihood'
    where_lklhd_columns_are = df.columns.get_level_values('coords') == 'likelihood'
    lklhd_columns = df.loc[:, where_lklhd_columns_are]
    # identify, in boolean array, rows where all columns have value less than threshold
    low_probability_timepoints = (lklhd_columns < threshold).all(axis=1)
    if np.any(low_probability_timepoints):
        # show the result
        plt.step(range(len(df)), low_probability_timepoints)
        plt.title(title)
        
        if save_figure:
            plt.savefig(os.path.join(save_dir, save_name))
        
        if show_figure:
            plt.show()
        else:
            plt.close()
    else:
        print(f"No time point where all body part likelihoods are below {threshold}!")
    
    return np.where(low_probability_timepoints)[0] if np.any(low_probability_timepoints) else []

def store_which_csv_to_truncate_and_when(csv_folder : list, 
                                         save_dir : str,
                                         save_csv_name : str,
                                         save_figure : bool=True,
                                         show_figure : bool=False,
                                         threshold : float=TRUNCATE_THRESHOLD):
    """
    Based on the existence of frames with all body parts predicted below threshold,
    identifies whether a given video needs truncation. If so, the first frame that 
    had low probability for all body parts, and its time stamp are found.
    Then, 1) mouse name, 2) the need to truncate, 3) start frame and 4) its time stamp
    are stored in a csv. Also, returns the obtained dataframe.

    :param list csv_folder: List of folders each holding csvs to analyze.
    :param str save_dir: Directory to which we save both the generated csv and figures.
    :param str save_csv_name: Name of csv to which we store truncation info.
    :param bool save_figure: Whether to save figures of when body parts with low
    probabilities occurred, defaults to True
    :param bool show_figure: Whether to show figures indicating when body parts with low
    probabilities occurred, defaults to False
    :param float threshold: Threshold determining a body part was predicted with low 
    likelihood below it, defaults to TRUNCATE_THRESHOLD

    :return pd.DataFrame df: Dataframe holding info that were generated.
    """
    column_names = ['mousename', 'need_to_truncate', 
                    'truncate_start_frame', 'truncate_start_timestamp']
    
    data = []
    for folder in csv_folder:
        for file in os.listdir(folder):
            # ignore non-csvs
            if not file.endswith('.csv'): continue
            csvfile = os.path.join(folder, file)
            mousename = get_mousename(csvfile)
            
            try:
                # render figures while calculating where body parts less than threshold are
                less_than_threshold_indices = visualize_low_probability_on_all_bodyparts(
                    csv_path=csvfile,
                    save_dir=save_dir,
                    save_name=f"{file}_low_prob.png",
                    title=f">{threshold} Probability For All Bodypart {mousename}",
                    threshold=threshold,
                    save_figure=save_figure, 
                    show_figure=show_figure
                    )
                
                # compute the minimum of the frames that have uncertain body parts
                if len(less_than_threshold_indices) != 0:
                    min_frame = np.min(less_than_threshold_indices)
                    min_timestamp = f"{min_frame // (40*60)}:{(min_frame % (40*60)) / 40}"
                else:
                    min_frame, min_timestamp = None, None
                
                # store result together with mouse name into data
                new_row = [mousename, len(less_than_threshold_indices) != 0, min_frame, min_timestamp]
                if new_row not in data:
                    data.append(new_row)
            except:
                print(f"File {file} wasn't successfully read as DLC csv; under {folder}...")
    
    # create a dataframe and store it as csv
    df = pd.DataFrame(data=data, columns=column_names)
    print("Saving csv holding info of whether to truncate...", end="")
    df.to_csv(os.path.join(save_dir, save_csv_name))
    print("SUCCESSFUL!")

    return df


def visualize_probability_average_on_all_bodyparts(csv_path : str,
                                                   save_dir : str,
                                                   title : str,
                                                   save_name : str,
                                                   save_figure : bool=True,
                                                   show_figure : bool=True):
    """
    Visualizes where the average probabilty of body parts for all time points.
    """

    df = read_dlc_csv_file(csv_path, include_scorer=False)
    # find all columns with entries pertaining to 'likelihood'
    where_lklhd_columns_are = df.columns.get_level_values('coords') == 'likelihood'
    lklhd_columns = df.loc[:, where_lklhd_columns_are]
    # compute the average and show
    lklhd_average = lklhd_columns.mean(axis=1)
    # show the result
    plt.step(range(len(df)), lklhd_average)
    plt.title(title)
    if save_figure:
        plt.savefig(os.path.join(save_dir, save_name))

    if show_figure:
        plt.show()
    else: 
        plt.close()


def truncate_given_csv_at_specified_timepoint(csv_path : str, 
                                              truncate_frame : int,
                                              outdir : str=None):
    """
    Truncate the passed csv's dataframe at 'truncate_frame', 
    saving the truncated version into outdir.

    :param str csv_path: Path to csv to be truncated.
    :param int truncate_frame: Frame at which we truncate the video. 
    The frame itself isn't contained!
    :param str outdir: Directory to save the truncated dataframe, 
    defaults to being stored in the same directory as the original.
    """
    if outdir == None:
        outdir = os.path.dirname(csv_path)
    filename = os.path.basename(csv_path)

    df = read_dlc_csv_file(csv_path)
    if df.shape[0] < truncate_frame:
        raise Exception(f"Csv has less timepoints than given truncated_frame {truncate_frame}...")
    
    truncated_df = df.loc[:truncate_frame-1, :]
    truncated_df.to_csv(os.path.join(outdir, filename.replace('.csv', f'{TRUNCATED_SUFFIX_NO_EXT}.csv')))
    
if __name__ == "__main__":

    SAVE_DIR = r"C:\Users\mashi\Desktop\temp\RaymondLab\DeepLabCut\COMPUTED\fig"

    ABOVE_CSV_DIR = r"C:\Users\mashi\Desktop\temp\Q175\csvs\iter4-2060000\pretrunc"
    CSV_DIRS = [
        os.path.join(ABOVE_CSV_DIR, 'filt'), 
        os.path.join(ABOVE_CSV_DIR, 'unfilt')
        ]
    OUTDIRS = [
        os.path.join(ABOVE_CSV_DIR.replace('pretrunc', 'trunc'), 'filt'), 
        os.path.join(ABOVE_CSV_DIR.replace('pretrunc', 'trunc'), 'unfilt')
    ]

    TRUNCATION_CSV = "Q175_video_need_to_truncate.csv"

    THRESHOLD = 0.8

    # df = store_which_csv_to_truncate_and_when(csv_folder=CSV_DIRS, 
    #                                           save_dir=SAVE_DIR,
    #                                           save_csv_name=TRUNCATION_CSV,
    #                                           save_figure=False,
    #                                           show_figure=False,
    #                                           threshold=THRESHOLD)
    
    df = pd.read_csv(r"C:\Users\mashi\Desktop\temp\Q175\Q175_video_need_to_truncate.csv")

    for indir, outdir in zip(CSV_DIRS, OUTDIRS):
        for file in os.listdir(indir):
            csvfile = os.path.join(indir, file)
            # we assume there's only one entry of the name mousename here
            for row_idx in np.where(df['mousename'] == get_mousename(csvfile))[0]:
                df_mousename = df['mousename'][row_idx]
                
                if df['need_to_truncate'][row_idx]:
                    df_truncate_frame = int(df['truncate_start_frame'][row_idx].item())
                    truncate_given_csv_at_specified_timepoint(csv_path=csvfile,
                                                              truncate_frame=df_truncate_frame,
                                                              outdir=outdir)
                else:
                    shutil.copyfile(csvfile, csvfile.replace('.csv', f'{TRUNCATED_SUFFIX_NO_EXT}.csv'))