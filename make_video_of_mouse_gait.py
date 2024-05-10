# Author: Akira Kudo
# Created: 2024/05/10
# Last Updated: 2024/05/10

import numpy as np
import pandas as pd

import sys

# I will learn about proper packaging and arrangement later...
sys.path.append(r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID")
sys.path.append(r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut")

from BSOID.bsoid_io.utils import read_BSOID_labeled_csv
from BSOID.feature_analysis_and_visualization.utils import find_runs, process_upper_and_lower_limit
from BSOID.label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

from DeepLabCut.utils_to_be_replaced_oneday import get_mousename
from DeepLabCut.temper_with_csv_and_hdf.do_make_video_from_dlc_and_png import extract_frames_and_construct_video_from_dataframe

# 1) Extract mouse locomotion by looking at sequences of motion from label
# 2) Within bouts of locomotion, filter and remove any dataframe value with movement
# 3) Render the whole as video.

MOVEMENT_THRESHOLD = 0.5
LOCOMOTION_LABEL = [38]
FPS = 40

def make_locomotion_paper_ink_video_from_dataframe_and_label(
        video_path : str,
        df : pd.DataFrame,
        label : np.ndarray,
        img_dir : str,
        output_dir : str,
        show_nonpaw : bool=False,
        length_limits : tuple=(None, None),
        num_runs : int=5,
        threshold : float=MOVEMENT_THRESHOLD,
        locomotion_labels : list=LOCOMOTION_LABEL, 
        fps : int=FPS
        ):
    """
    Create locomotion video in the paper-ink paradigm manner, 
    using the given dataframe and label.

    :param str video_path: Path to video.
    :param pd.DataFrame df: DataFrame holding DLC data.
    :param np.ndarray label: BSOID label for the DLC data. 
    :param str img_dir: Directory holding frames extracted from video.
    :param str output_dir: Directory where we output the generated video.
    :param bool show_nonpaw: Whether to show non-paw body parts, 
    defaults to False
    :param tuple length_limits: Lower & upper limits in length for locomotion
    snippets to use to generate videos, defaults to (None, None)
    :param int num_runs: Number of locomotion runs to render as video, defaults to 5
    :param float threshold: Threshold separating locomotion movement from 
    paw rest, defaults to MOVEMENT_THRESHOLD
    :param list locomotion_labels: Integer labels corresponding to locomotion groups, 
    defaults to LOCOMOTION_LABEL (38, for YAC128 network).
    :param int fps: Frame-per-second for generated video, defaults to FPS
    """
    
    if len(locomotion_labels) < 1: 
        raise Exception("At least 1 locomotion label integer must be specified...")
    
    len_lowlim, len_highlim = process_upper_and_lower_limit(length_limits)

    # merge all locomotion labels into one category
    for loc_lbl in locomotion_labels:
        if loc_lbl != locomotion_labels[0]:
            label[label == loc_lbl] = locomotion_labels[0]
    # remove entries in df where movement of 'paws' is above threshold
    for bpt in np.unique(df.columns.get_level_values('bodyparts')):
        if 'paw' in bpt:
            x, y = df[bpt, 'x'].to_numpy(), df[bpt, 'y'].to_numpy()
            movement = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            movement = np.insert(movement, 0, 0) # pad first position as no movement
            df.loc[movement > threshold, [bpt]] = np.nan
        # completely remove entries for parts like snout and tailbase
        else:
            if not show_nonpaw:
                df.loc[:, [bpt]] = np.nan
    # remove entries in df that aren't locomotion bouts
    df.loc[label != locomotion_labels[0], :] = np.nan

    # extract runs of locomotion
    run_values, run_starts, run_lengths = find_runs(label)
    run_values, run_starts, run_lengths = np.array(run_values), np.array(run_starts), np.array(run_lengths)

    # generate videos for 'num_runs' locomotion bout
    locomotion_run_indices = np.where(run_values == locomotion_labels[0])[0]
    count = 0
    for run_idx in locomotion_run_indices:
        if count > num_runs: break

        start, length = run_starts[run_idx], run_lengths[run_idx]
        end = start + length - 1
        # check for length of bout and skip if not matching
        if length < len_lowlim or len_highlim < length:
            continue

        # generate the video!
        mousename = get_mousename(video_path)
        extract_frames_and_construct_video_from_dataframe(
            video_path=video_path, dlc_df=df, 
            img_dir=img_dir, output_dir=output_dir, 
            start=start, end=end, fps=fps, img_name=None, 
            output_name=f"{mousename}_paperNink_{start}to{end}_{fps}fps.mp4", 
            trailpoints=length+1
        )
        count += 1

if __name__ == "__main__":
    VIDEO_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\\" + \
        "20220228223808_320151_m1_openfield.mp4"

    BSOID_CSV_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr082024\CSV files\BSOID\\" + \
        "Apr-08-2024labels_pose_40Hz20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"

    IMG_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\extracted"
    OUTPUT_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\generated"

    NUM_RUNS = 3
    LOCOMOTION_LABEL = [29,30]
    FPS = 10

    LEN_LIMS = (60, None)

    label, df = read_BSOID_labeled_csv(BSOID_CSV_PATH)
    label = filter_bouts_smaller_than_N_frames(label=label, n=5)

    make_locomotion_paper_ink_video_from_dataframe_and_label(
        video_path=VIDEO_PATH,
        df=df,
        label=label,
        img_dir=IMG_DIR, output_dir=OUTPUT_DIR, num_runs=NUM_RUNS,
        threshold=MOVEMENT_THRESHOLD,
        locomotion_labels=LOCOMOTION_LABEL, 
        fps=FPS,
        length_limits=LEN_LIMS,
        show_nonpaw=True
        )