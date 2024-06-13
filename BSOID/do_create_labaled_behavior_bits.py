# Author: Akira Kudo
# Created: 2024/03/21
# Last Updated: 2024/06/10

import os

from bsoid_io.utils import read_BSOID_labeled_features
from feature_analysis_and_visualization.utils import get_mousename
from label_behavior_bits.create_labeled_behavior_bits import create_labeled_behavior_bits, extract_label_from_labeled_csv, filter_bouts_smaller_than_N_frames

def create_behavior_bits_from_filtered_label(
        labels, 
        crit, 
        counts,
        output_fps, 
        frame_dir,
        output_path,
        data_csv_path,
        video_path,
        dotsize,
        colormap,
        bodyparts2plot,
        trailpoints,
        choose_from_top_or_random, 
        n : int
        ):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors (~300ms is advised)
    :param counts: scalar, number of generated examples (~5 is advised)
    :param output_fps: integer, frame per second for the output video
    :param video_path: string, path to video from which to extract additional vid images
    :param frame_dir: string, directory to where you extracted vid images
    :param output_path: string, directory to where you want to store short video examples

    :param choose_from_top_or_random: Whether to choose 'counts' groups at random or from the 
    top N 'counts' in terms of length. If not "random", chooses the top 'counts'.
    :param int n: The length of the filtering width. Uses filter_bouts_smaller_than_N_frames
    to do the filtering (check out its specification for details!)
    """
    print("Pre-filtering...")
    filtered = filter_bouts_smaller_than_N_frames(labels, n=n)
    print("Done!")

    create_labeled_behavior_bits(labels=filtered, 
                                 crit=crit,
                                 counts=counts,
                                 output_fps=output_fps, 
                                 frame_dir=frame_dir,
                                output_path=output_path,
                                data_csv_path=data_csv_path,
                                video_path=video_path,
                                dotsize=dotsize,
                                colormap=colormap,
                                bodyparts2plot=bodyparts2plot,
                                trailpoints=trailpoints,
                                choose_from_top_or_random=choose_from_top_or_random)

if __name__ == "__main__":
    FPS = 40
    FILTERING_NOISE_MAX_LENGTH = 5 # max length of noise filtered via filter_bouts_smaller_than_N_frames
    MIN_DESIRED_BOUT_LENGTH = 500
    COUNTS = 2
    OUTPUT_FPS = 20
    TRAILPOINTS = 0
    TOP_OR_RANDOM = "top" #"random"

    DOTSIZE = 5
    COLORMAP = "rainbow" # obtained from config.yaml on DLC side
    # we exclude "belly" as it isn't used to classify in this B-SOID
    BODYPARTS = ["snout", "rightforepaw", "leftforepaw", 
                 "righthindpaw", "lefthindpaw",  "tailbase"] 
    
    STOP_UPON_MISSING_VIDEO = False
    
    MOUSEGROUP = "Q175" if False else ("Q175_Black" if True else "Q175")
    
    #####DEFINING OUTPUT FOLDER#####
    if MOUSEGROUP == "Q175":
        OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr082024\labeled_behavior_bits\filterlen5\allmice"
        VIDEO_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos"
    elif MOUSEGROUP == "Q175_Black":
        OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr082024\labeled_behavior_bits\filterlen5\black"
        VIDEO_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\videos\blackmice"
    elif MOUSEGROUP == "YAC128":
        OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\Feb232023\labeled_behavior_bits\filterlen5"
        VIDEO_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos"

    #####DEFINING OTHER IMPORTANT FOLDERS#####
    FRAME_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\results\pngs"

    OUTPUT_PATH = OUTPUT_FOLDER

    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    # we are analyzing a whole directory
    # Q175
    if MOUSEGROUP == "Q175":
        labelfile_folders = [
            # HD_filt labeled feature
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\allcsv_2024_05_16_Akira\HD_filt",
            # WT_filt labeled feature
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\allcsv_2024_05_16_Akira\WT_filt"
        ]

        dlcfile_folders = [    
            # HD_filt csvs
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\allcsv_2024_05_16_Akira\HD_filt",
            # WT_filt csvs
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\allcsv_2024_05_16_Akira\WT_filt"
        ]
    elif MOUSEGROUP == "Q175_Black":
        # TO BE FIXED START
        labelfile_folders = [
            # HD_filt labeled feature
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\black\it0-2000k\HD_filt",
            # WT_filt labeled feature
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\black\it0-2000k\WT_filt"
        ]

        dlcfile_folders = [    
            # HD_filt csvs
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\black\it0-2000k\HD_filt",
            # WT_filt csvs
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\black\it0-2000k\WT_filt"
        ]
        # TO BE FIXED END
    elif MOUSEGROUP == "YAC128":
        labelfile_folders = [
            # HD_filt labeled feature
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\allcsv_2024_05_16_Akira\HD_filt",
            # WT_filt labeled feature
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\allcsv_2024_05_16_Akira\WT_filt"
        ]

        dlcfile_folders = [    
            # HD_filt csvs
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\csv\allcsv_2024_05_16_Akira\HD_filt",
            # WT_filt csvs
            r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\csv\allcsv_2024_05_16_Akira\WT_filt"
        ]

    labelfiles = [os.path.join(folder, file) for folder in labelfile_folders 
                for file in os.listdir(folder) if file.endswith(".csv")]
    labelfiles.sort() # by sorting, I hope to establish the same ordering as dlcfiles
    
    dlcfiles = [os.path.join(folder, file) for folder in dlcfile_folders 
                for file in os.listdir(folder) if file.endswith(".csv")]
    dlcfiles.sort() # by sorting, I hope to establish the same ordering as labelfiles

    if len(labelfiles) != len(dlcfiles): 
        raise Exception("labelfiles must be the same length as dlcfiles, but is " + 
                        f"{len(labelfiles)} and {len(dlcfiles)} respectively... ")
    
    for lblfl, dlcfl in zip(labelfiles, dlcfiles):
        if get_mousename(lblfl) != get_mousename(dlcfl):
            raise Exception("lblfl must be the same mouse as dlcfl, but is different:" + 
                        f"{get_mousename(lblfl)} and {get_mousename(dlcfl)} respectively... ")
        
        # find the video with matching mouse name
        video_found = False
        for file in os.listdir(VIDEO_FOLDER):
            if get_mousename(file) == get_mousename(lblfl):
                video_found = True
                video_path = os.path.join(VIDEO_FOLDER, file)
                break
        if not video_found:
            if STOP_UPON_MISSING_VIDEO:
                raise Exception(f"Video for mouse {get_mousename(lblfl)} not found...")
            else:
                print(f"Video for mouse {get_mousename(lblfl)} not found...")

        labels = read_BSOID_labeled_features(lblfl)[0]

        framedir = os.path.join(FRAME_DIR, get_mousename(lblfl))
        if not os.path.exists(framedir):
            os.mkdir(framedir)

        outputpath = os.path.join(OUTPUT_PATH, get_mousename(lblfl))
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
        else:
            continue

        create_behavior_bits_from_filtered_label(
            labels=labels,
            crit=MIN_DESIRED_BOUT_LENGTH / FPS, 
            counts=COUNTS,
            output_fps=OUTPUT_FPS, 
            frame_dir=framedir,
            output_path=outputpath,
            video_path=video_path,
            data_csv_path=dlcfl,
            dotsize=DOTSIZE,
            colormap=COLORMAP,
            bodyparts2plot=BODYPARTS,
            trailpoints=TRAILPOINTS,
            choose_from_top_or_random=TOP_OR_RANDOM,
            n=FILTERING_NOISE_MAX_LENGTH
            )