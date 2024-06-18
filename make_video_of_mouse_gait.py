# Author: Akira Kudo
# Created: 2024/05/10
# Last Updated: 2024/06/18

import numpy as np
import pandas as pd

import sys

# I will learn about proper packaging and arrangement later...
sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID")
sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut")

from BSOID.bsoid_io.utils import read_BSOID_labeled_csv
from BSOID.feature_analysis_and_visualization.visualization.visualize_mouse_gait import filter_nonpawrest_motion, select_N_locomotion_sequences
from BSOID.label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

from DeepLabCut.dlc_io.utils import read_dlc_csv_file
from DeepLabCut.temper_with_csv_and_hdf.data_filtering.identify_paw_noise import identify_bodypart_noise_by_impossible_speed
from DeepLabCut.temper_with_csv_and_hdf.data_filtering.filter_based_on_boolean_array import filter_based_on_boolean_array
from DeepLabCut.temper_with_csv_and_hdf.do_make_video_from_dlc_and_png import extract_frames_and_construct_video_from_dataframe
from DeepLabCut.utils_to_be_replaced_oneday import get_mousename


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
        filter_size : int=5,
        threshold : float=MOVEMENT_THRESHOLD,
        locomotion_labels : list=LOCOMOTION_LABEL, 
        fps : int=FPS,
        average_pawrest : bool=True
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
    :param int filter_size: Any bout smaller than this size is replaced by its neighbors
    using filter_bouts_smaller_than_N_frames, defaults to 5.
    :param float threshold: Threshold separating locomotion movement from 
    paw rest, defaults to MOVEMENT_THRESHOLD
    :param list locomotion_labels: Integer labels corresponding to locomotion groups, 
    defaults to LOCOMOTION_LABEL (38, for YAC128 network).
    :param int fps: Frame-per-second for generated video, defaults to FPS
    :param bool average_pawrest: Whether to make each continuous paw rest that is identified
    averaged, so that a single point is identified. E.g. if the X coord is: 
    [NaN, 3, 3.2, 3.1, NaN, 5, 4.8, NaN]..., we average this to be:
    [NaN, 3.1, 3.1, 3.1, NaN, 4.9, 4.9, NaN].
    """
    
    if len(locomotion_labels) < 1: 
        raise Exception("At least 1 locomotion label integer must be specified...")
    
    # merge all locomotion labels into one category
    for loc_lbl in locomotion_labels:
        if loc_lbl != locomotion_labels[0]:
            label[label == loc_lbl] = locomotion_labels[0]
    # then filter any bout smaller than filter_size
    label = filter_bouts_smaller_than_N_frames(label=label, n=filter_size)
    
    # also filter non-paw rest motion from the DLC data
    df, _ = filter_nonpawrest_motion(
        df=df, label=label, show_nonpaw=show_nonpaw, threshold=threshold, 
        locomotion_labels=locomotion_labels, average_pawrest=average_pawrest
        )
    
    # extract runs of locomotion
    starts, ends, lengths = select_N_locomotion_sequences(
        label=label, N=num_runs, locomotion_labels=locomotion_labels[0], 
        length_limits=length_limits
        )
    
    mousename = get_mousename(video_path)

    # generate videos for 'num_runs' locomotion bout
    for start, end, length in zip(starts, ends, lengths):
        # generate the video!
        
        extract_frames_and_construct_video_from_dataframe(
            video_path=video_path,   dlc_df=df, 
            img_dir=img_dir, output_dir=output_dir, 
            start=start, end=end, fps=fps, img_name=None, 
            output_name=f"{mousename}_paperNink_{start}to{end}_{fps}fps.mp4", 
            trailpoints=length+1
        )

if __name__ == "__main__":
    VIDEO_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\\" + \
        "20220228223808_320151_m1_openfield.mp4"
    VIDEO_PATH2 = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos\\" + \
        "20220213051553_326787_m2.mp4"
    
    BSOID_CSV_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr082024\CSV files\BSOID\\" + \
        "Apr-08-2024labels_pose_40Hz20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"
    
    DLC_CSV_PATH2 = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\csvs\csv_1stTermPres\mix-it1-1030k-it3-2100k\WT_unfilt\\" + \
        "20220213051553_326787_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
    BSOID_LABEL_PATH2 = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\feats_labels\\" + \
        "Feb-23-2023_20220213051553_326787_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000_labels.npy"

    IMG_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\extracted"
    IMG_DIR2 = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos\extracted"
    OUTPUT_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\generated"
    OUTPUT_DIR2 = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos\generated"

    NUM_RUNS = float("inf")
    FPS = 10

    BODYPARTS = [
        "snout", 
        "rightforepaw", "leftforepaw",
        "righthindpaw", "lefthindpaw",
        "tailbase", "belly"
    ]

    LEN_LIMS = (60, None)

    def filter_label_and_dataframe(label : np.ndarray, 
                                   df : pd.DataFrame):
        # then filter paw noises based on impossible speed
        noise_df = identify_bodypart_noise_by_impossible_speed(bpt_data=df,
                                                               bodyparts=BODYPARTS,
                                                               start=0, 
                                                               end=df.shape[0], # number of rows
                                                               savedir=None, 
                                                               save_figure=False, 
                                                               show_figure=False)
        filt_df = df 
        for bpt in BODYPARTS:
            bpt_bool_arr = noise_df[(bpt, "loc_wrng")].to_numpy()

            filt_df = filter_based_on_boolean_array(
                bool_arr=bpt_bool_arr, df=filt_df, bodyparts=[bpt], filter_mode="linear"
            )

        filt_label = label
        
        return filt_label, filt_df
    
    # for video 1 - Q175
    if True:
        LOCOMOTION_LABEL = [29,30]
        
        label, df = read_BSOID_labeled_csv(BSOID_CSV_PATH)
        filt_lbl, filt_df = filter_label_and_dataframe(label, df)
        
        make_locomotion_paper_ink_video_from_dataframe_and_label(
            video_path=VIDEO_PATH,
            df=filt_df,
            label=filt_lbl,
            img_dir=IMG_DIR, output_dir=OUTPUT_DIR, num_runs=NUM_RUNS,
            filter_size=5,
            threshold=MOVEMENT_THRESHOLD,
            locomotion_labels=LOCOMOTION_LABEL, 
            fps=FPS,
            length_limits=LEN_LIMS,
            show_nonpaw=False,
            average_pawrest=True
            )
    
    # for video 2 - YAC128
    if False:
        LOCOMOTION_LABEL = [38]

        movement_threshold = 0.5

        label, df = np.load(BSOID_LABEL_PATH2), read_dlc_csv_file(dlc_path=DLC_CSV_PATH2)
        label = np.insert(label, 0, [label[0].item()] * 3)
        filt_lbl, filt_df = filter_label_and_dataframe(label, df)
        
        make_locomotion_paper_ink_video_from_dataframe_and_label(
            video_path=VIDEO_PATH2,
            df=filt_df,
            label=filt_lbl,
            img_dir=IMG_DIR2, output_dir=OUTPUT_DIR2, num_runs=NUM_RUNS,
            filter_size=5,
            threshold=movement_threshold,
            locomotion_labels=LOCOMOTION_LABEL, 
            fps=FPS,
            length_limits=LEN_LIMS,
            show_nonpaw=False
            )