# Author: Akira Kudo
# Created: 2024/05/10
# Last Updated: 2024/06/18

import os
import sys

import numpy as np
import pandas as pd

# I will learn about proper packaging and arrangement later...
sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID")
sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut")

from BSOID.bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
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
FPS = 40
LOCOMOTION_LABEL = [29,30]

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

def main(video_path : str,
        dlc_path : str,
        label_path : str,
        img_dir : str,
        output_dir : str,
        show_nonpaw : bool=False,
        length_limits : tuple=(None, None),
        num_runs : int=5,
        filter_size : int=5,
        threshold : float=MOVEMENT_THRESHOLD,
        locomotion_labels : list=LOCOMOTION_LABEL, 
        fps : int=FPS,
        average_pawrest : bool=True):
    
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
    
    # read csvs
    label, _ = read_BSOID_labeled_features(csv_path=label_path)
    df = read_dlc_csv_file(dlc_path=dlc_path, include_scorer=False)

    # TRUNCATE THE DLC DATA
    if len(label) < df.shape[0]:
        print(f"Label ({len(label)} long) is shorter than the DLC data ({df.shape[0]} long) " + 
            ", but we know the DLC data should match the labels at the beginning.")
        print("We will hence truncate the DLC data!")
        df = df.loc[:len(label)-1, :]

    filt_lbl, filt_df = filter_label_and_dataframe(label, df)

    make_locomotion_paper_ink_video_from_dataframe_and_label(
        video_path=video_path,
        df=filt_df,
        label=filt_lbl,
        img_dir=img_dir, output_dir=output_dir, num_runs=num_runs,
        filter_size=filter_size,
        threshold=threshold,
        locomotion_labels=locomotion_labels, 
        fps=fps,
        length_limits=length_limits,
        show_nonpaw=show_nonpaw,
        average_pawrest=average_pawrest
        )

if __name__ == "__main__":
    NUM_RUNS = float("inf")
    FPS = 10

    BODYPARTS = [
        "snout", 
        "rightforepaw", "leftforepaw",
        "righthindpaw", "lefthindpaw",
        "tailbase", "belly"
    ]

    LEN_LIMS = (60, None)

    def find_corresponding_video(video_folder : str, mousename : str):
        # standardize format
        mousename = get_mousename(mousename)

        for file in os.listdir(video_folder):
            if not file.endswith(".mp4"): continue
            video_mouse = get_mousename(file)
            if video_mouse == mousename:
                return os.path.join(video_folder, file)
        raise Exception(f"Video for {mousename} not found in: {video_folder}...")
    
    # for Q175
    if True:        
        IMG_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\extracted"
        OUTPUT_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\generated\pawInk"

        ABOVE_LABEL_CSVS = [
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\allcsv_2024_06_20_Akira\HD_filt",
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\allcsv_2024_06_20_Akira\WT_filt"
        ]
        LABEL_CSVS_PATHS = []; [LABEL_CSVS_PATHS.extend([os.path.join(csv_folder, file) 
                                             for file in os.listdir(csv_folder) if file.endswith('.csv')]) 
                          for csv_folder in ABOVE_LABEL_CSVS]
        
        ABOVE_DLC_CSVS = [
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\allcsv_2024_06_20_Akira\HD_filt",
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\allcsv_2024_06_20_Akira\WT_filt"
        ]
        LABEL_DLC_PATHS = []; [LABEL_DLC_PATHS.extend([os.path.join(csv_folder, file)
                                             for file in os.listdir(csv_folder) if file.endswith('.csv')]) 
                          for csv_folder in ABOVE_DLC_CSVS]
        
        LOCOMOTION_LABEL = [29,30]

        ABOVE_VIDEOS = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos"
        ABOVE_VIDEOS2 = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\blackmice"
        
        for lbl, dlc in zip(LABEL_CSVS_PATHS, LABEL_DLC_PATHS):
            # check that the two files point the same mouse
            if get_mousename(lbl) != get_mousename(dlc):
                raise Exception("Provided pair of label & dlc csvs aren't for the same mouse: \n" + 
                                f" - {os.path.basename(lbl)} \n" + 
                                f" - {os.path.basename(dlc)} \n")
            # find the corresponding video
            mousename = get_mousename(lbl)
            try:
                video_path = find_corresponding_video(video_folder=ABOVE_VIDEOS, mousename=mousename)
            except:
                video_path = find_corresponding_video(video_folder=ABOVE_VIDEOS2, mousename=mousename)
            
            print(f"Analyzing mouse: {mousename}:")
            print(f"- {os.path.basename(lbl)}")
            print(f"- {os.path.basename(dlc)}")
            print(f"- {os.path.basename(video_path)}")

            # setup outputdir based on the mouse of interest
            outputdir = os.path.join(OUTPUT_DIR, mousename)

            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
        
            main(video_path=video_path,
                dlc_path=dlc,
                label_path=lbl,
                img_dir=IMG_DIR, output_dir=outputdir, num_runs=NUM_RUNS,
                filter_size=5,
                threshold=MOVEMENT_THRESHOLD,
                locomotion_labels=LOCOMOTION_LABEL, 
                fps=FPS,
                length_limits=LEN_LIMS,
                show_nonpaw=False,
                average_pawrest=True
                )
    
    # for video 2 - YAC128
    if True:
        IMG_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos\extracted"
        OUTPUT_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos\generated\pawInk"

        ABOVE_LABEL_CSVS = [
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\allcsv_2024_05_16_Akira\HD_filt", 
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\allcsv_2024_05_16_Akira\WT_filt"
        ]
        LABEL_CSVS_PATHS = []; [LABEL_CSVS_PATHS.extend([os.path.join(csv_folder, file) 
                                             for file in os.listdir(csv_folder) if file.endswith('.csv')]) 
                          for csv_folder in ABOVE_LABEL_CSVS]
        
        ABOVE_DLC_CSVS = [
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\csv\allcsv_2024_05_16_Akira\filt\HD_filt",
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\csv\allcsv_2024_05_16_Akira\filt\WT_filt"
        ]
        LABEL_DLC_PATHS = []; [LABEL_DLC_PATHS.extend([os.path.join(csv_folder, file)
                                             for file in os.listdir(csv_folder) if file.endswith('.csv')]) 
                          for csv_folder in ABOVE_DLC_CSVS]
        
        ABOVE_VIDEOS = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos"

        LOCOMOTION_LABEL = [38]
        
        for lbl, dlc in zip(LABEL_CSVS_PATHS, LABEL_DLC_PATHS):
            # check that the two files point the same mouse
            if get_mousename(lbl) != get_mousename(dlc):
                raise Exception("Provided pair of label & dlc csvs aren't for the same mouse: \n" + 
                                f" - {os.path.basename(lbl)} \n" + 
                                f" - {os.path.basename(dlc)} \n")
            # find the corresponding video
            mousename = get_mousename(lbl)
            video_path = find_corresponding_video(video_folder=ABOVE_VIDEOS, mousename=mousename)
            
            print(f"Analyzing mouse: {mousename}:")
            print(f"- {os.path.basename(lbl)}")
            print(f"- {os.path.basename(dlc)}")
            print(f"- {os.path.basename(video_path)}")

            # setup outputdir based on the mouse of interest
            outputdir = os.path.join(OUTPUT_DIR, mousename)

            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
        
            main(video_path=video_path,
                dlc_path=dlc,
                label_path=lbl,
                img_dir=IMG_DIR, output_dir=outputdir, num_runs=NUM_RUNS,
                filter_size=5,
                threshold=MOVEMENT_THRESHOLD,
                locomotion_labels=LOCOMOTION_LABEL, 
                fps=FPS,
                length_limits=LEN_LIMS,
                show_nonpaw=False,
                average_pawrest=True
                )