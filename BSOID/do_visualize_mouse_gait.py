# Author: Akira Kudo
# Created: 2024/05/17
# Last Updated: 2024/06/26

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bsoid_io.utils import read_BSOID_labeled_features
from feature_analysis_and_visualization.visualization.visualize_mouse_gait import visualize_locomotion_stats, visualize_mouse_gait_speed, visualize_stepsize_in_locomotion, visualize_stepsize_in_locomotion_in_multiple_mice, visualize_stepsize_in_locomotion_in_single_mouse, visualize_stepsize_in_locomotion_in_mice_groups
from label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

BINSIZE = 20

# YAC128
if False:
    LABEL_CSV = os.path.join(
        r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\allcsv_2024_05_16_Akira\WT_filt", 
        "Feb-23-2023_20220213051553_326787_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000_filtered_labeled_features.csv"
        )
    DLC_CSV = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\csv\allcsv_2024_05_16_Akira\unfilt\WT_unfilt\20220213051553_326787_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
    LOCOMOTION_LABEL = [38]

    label, _ = read_BSOID_labeled_features(LABEL_CSV)

    LENGTH_LIMITS = (60, None)
    PLOT_N_RUNS = float("inf")

    ABOVE_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\fig\pawInk"

    WT_MICE = [
        # Male
        "312172m2", "312172m5", "326787m1", "326787m2", "349412m5", "351201m3", "374451m1",
        # Female
    ]

    HD_MICE = [
        # Male 
        "308535m1", "312152m2", "312153m2", "312172m3", "315955m1", "315955m2", "326787m3",
        "326787m5", "392607m1", "395035m1", "395035m3"
        # Female
    ]

    MALE_MICE = [
        # WT
        "312172m2", "312172m5", "326787m1", "326787m2", "349412m5", "351201m3", "374451m1",
        # HD
        "308535m1", "312152m2", "312153m2", "312172m3", "315955m1", "315955m2", "326787m3",
        "326787m5", "392607m1", "395035m1", "395035m3"
    ]

    FEMALE_MICE = [
        # WT
        # HD
        ]

else: # Q175
    LABEL_NPY = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\results\feats_labels\\" + \
        "Apr-08-2024_20230113135134_372301_m3_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_2060000_labels.npy"
    DLC_CSV = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\allcsv_2024_05_16_Akira\filt\HD_filt\20230113135134_372301_m3_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_2060000_filtered.csv"
    LOCOMOTION_LABEL = [29,30]

    label = np.load(LABEL_NPY)

    LENGTH_LIMITS = (60, None)
    PLOT_N_RUNS = float("inf")

    ABOVE_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\fig\pawInk"

    WT_MICE = [
        # Male Brown
        "316367m2", "320151m1", "320151m2" "320151m3",  
        "363453m1", 
        # Female Brown
        "301524f1", "301524f23", "301532f2", 
        "312142f1", "363451f5",
        # Male Black
        # Female Black
        "301525f2", "312142f2"
    ]

    HD_MICE = [
        # Male Brown
        "362816m1", "372301m1", "372301m3", 
        # Female Brown
        # 301533f3 << CAN'T INCLUDE - OPEN WOUND
        "327245f1", "327245f2", "327245f5", "363451f1",
        # Male Black
        "328147m1", "355988m7",
        # Female Black
        "312142f3", "361212f5"
    ]

    MALE_MICE = [
        # Brown WT
        "316367m2", "320151m1", "320151m2", "320151m3",
        "363453m1", 
        # Brown HD
        "362816m1", "372301m1", "372301m3", 
        # Black WT
        # Black HD
        "328147m1", "355988m7",
    ]

    FEMALE_MICE = [
        # Brown WT
        "301524f1", "301524f23", "301532f2", "312142f1",
        "363451f5",
        # Brown HD
        # 301533f3 << CAN'T INCLUDE - OPEN WOUND
        "327245f1", "327245f2", "327245f5", "363451f1",
        # Black WT
        "301525f2", "312142f2", 
        # Black HD
        "312142f3", "361212f5"
    ]
        

if False:
    print("Pre-filtering!")
    visualize_locomotion_stats(label=label,
                                figure_name=None,
                                save_path=None,
                                interval=40*60*5,
                                locomotion_label=LOCOMOTION_LABEL,
                                save_figure=False,
                                show_figure=True)

    df = pd.read_csv(DLC_CSV, header=[1,2], index_col=0)
    print(df)

    visualize_mouse_gait_speed(df=df, 
                            label=label, 
                            bodyparts=['snout', 
                                        'rightforepaw', 'leftforepaw', 
                                        'righthindpaw', 'lefthindpaw', 
                                        'tailbase', 'belly'], 
                                length_limits=LENGTH_LIMITS,
                                locomotion_label=LOCOMOTION_LABEL, 
                                plot_N_runs=PLOT_N_RUNS)

    # merge all locomotion labels into one category
    for loc_lbl in LOCOMOTION_LABEL:
            if loc_lbl != LOCOMOTION_LABEL[0]:
                label[label == loc_lbl] = LOCOMOTION_LABEL[0]
    # then filter any bout smaller than filter_size    
    filt_label = filter_bouts_smaller_than_N_frames(label=label, n=5)

    print("Post-filtering!")
    visualize_locomotion_stats(label=filt_label,
                                figure_name=None,
                                save_path=None,
                                interval=40*60*5,
                                locomotion_label=LOCOMOTION_LABEL,
                                save_figure=False,
                                show_figure=True)

    visualize_mouse_gait_speed(df=df, 
                            label=filt_label, 
                            bodyparts=['snout', 
                                        'rightforepaw', 'leftforepaw', 
                                        'righthindpaw', 'lefthindpaw', 
                                        'tailbase', 'belly'], 
                                length_limits=LENGTH_LIMITS,
                                locomotion_label=LOCOMOTION_LABEL, 
                                plot_N_runs=PLOT_N_RUNS)

# visualize_stepsize_in_locomotion_in_single_mouse
if True:
    # visualize step size in each mice - requires the 'paw rest distance over time' files
    if False:
        print("Processing single mice!")
        for mouse_idx, mousename in enumerate(os.listdir(ABOVE_FOLDER)):
            
            fullpath = os.path.join(ABOVE_FOLDER, mousename)
            # skip any non-directory
            if os.path.isfile(fullpath): continue
            
            print(f"Processing: {mousename}!")
            for file in os.listdir(fullpath):
                if 'pawRestDistanceOverTime' in file:
                    even_fullpath = os.path.join(fullpath, file)
                    
                    visualize_stepsize_in_locomotion_in_single_mouse(
                        stepsize_yaml=even_fullpath,
                        title=f"Distance Between Consecutive Rests ({mousename})",
                        savedir=fullpath,
                        savename=f"distanceBetweenConsecutiveRests_{mousename}",
                        binsize=BINSIZE,
                        show_figure=False,
                        save_figure=True
                    )
                    break
        print("SINGLE MICE DONE!")

    if True:
        YAMLS = [os.path.join(ABOVE_FOLDER, folder, yaml) 
                for folder in [os.path.join(ABOVE_FOLDER, fol) 
                               for fol in os.listdir(ABOVE_FOLDER) 
                               if os.path.isdir(os.path.join(ABOVE_FOLDER, fol))]
                for yaml in os.listdir(os.path.join(ABOVE_FOLDER, folder))
                if yaml.endswith(".yaml")]
        
        

        WT_YAMLS = [os.path.join(ABOVE_FOLDER, folder, yaml) 
                    for folder in [os.path.join(ABOVE_FOLDER, fol) 
                                for fol in os.listdir(ABOVE_FOLDER) 
                                if os.path.isdir(os.path.join(ABOVE_FOLDER, fol)) and 
                                fol in WT_MICE]
                    for yaml in os.listdir(os.path.join(ABOVE_FOLDER, folder))
                    if yaml.endswith(".yaml")]
        
        HD_YAMLS = [os.path.join(ABOVE_FOLDER, folder, yaml) 
                    for folder in [os.path.join(ABOVE_FOLDER, fol) 
                                for fol in os.listdir(ABOVE_FOLDER) 
                                if os.path.isdir(os.path.join(ABOVE_FOLDER, fol)) and 
                                fol in HD_MICE]
                    for yaml in os.listdir(os.path.join(ABOVE_FOLDER, folder))
                    if yaml.endswith(".yaml")]
        
        MALE_YAMLS = [os.path.join(ABOVE_FOLDER, folder, yaml) 
                    for folder in [os.path.join(ABOVE_FOLDER, fol) 
                                for fol in os.listdir(ABOVE_FOLDER) 
                                if os.path.isdir(os.path.join(ABOVE_FOLDER, fol)) and 
                                fol in MALE_MICE]
                    for yaml in os.listdir(os.path.join(ABOVE_FOLDER, folder))
                    if yaml.endswith(".yaml")]
        
        FEMALE_YAMLS = [os.path.join(ABOVE_FOLDER, folder, yaml) 
                    for folder in [os.path.join(ABOVE_FOLDER, fol) 
                                for fol in os.listdir(ABOVE_FOLDER) 
                                if os.path.isdir(os.path.join(ABOVE_FOLDER, fol)) and 
                                fol in FEMALE_MICE]
                    for yaml in os.listdir(os.path.join(ABOVE_FOLDER, folder))
                    if yaml.endswith(".yaml")]
        
        if False:
            YAMLGROUPS = [YAMLS, HD_YAMLS, WT_YAMLS, MALE_YAMLS, FEMALE_YAMLS]
            if "Q175" in ABOVE_FOLDER:
                GROUPNAMES = ["Q175-B6", "Q175", "B6", "Male Q175-B6", "Female Q175-B6"]
            elif "YAC128" in ABOVE_FOLDER:
                GROUPNAMES = ["YAC128-FVB", "YAC128", "FVB", "Male YAC128-FVB", "Female YAC128-FVB"]

            print("Processing mice individually per group!")
            for yaml_group, groupname in zip(YAMLGROUPS, GROUPNAMES):
                print(f"Processing group: {groupname}!")
                visualize_stepsize_in_locomotion_in_multiple_mice(
                    yamls=yaml_group,
                    title=f"Frequency of Step Size Locomotion In {groupname} Mice",
                    savedir=ABOVE_FOLDER,
                    savename=f"VisualizeStepsizeInLocomotionIn{groupname.replace(' ', '')}",
                    binsize=BINSIZE,
                    show_figure=False,
                    save_figure=True
                )
            
            print("PROCESSING MICE INDIVIDUALLY PER GROUP DONE!")
        
        if True:
            # YAC128
            if "YAC128" in ABOVE_FOLDER:
                if True:
                    GROUPNAMES = ["YAC128", "FVB"]
                    YAMLGROUPS = [HD_YAMLS, WT_YAMLS]
                    
                else:
                    GROUPNAMES = ["Male YAC128-FVB", "Female YAC128-FVB"]
                    YAMLGROUPS = [MALE_YAMLS, FEMALE_YAMLS]
            
            # Q175
            elif "Q175" in ABOVE_FOLDER:
                if True:
                    GROUPNAMES = ["Q175", "B6"]
                    YAMLGROUPS = [HD_YAMLS, WT_YAMLS]
                else:
                    GROUPNAMES = ["Male Q175-B6", "Female Q175-B6"]
                    YAMLGROUPS = [MALE_YAMLS, FEMALE_YAMLS]
            
            print("Processing mice grouped together!")
            visualize_stepsize_in_locomotion_in_mice_groups(
                yamls_groups=YAMLGROUPS,
                groupnames=GROUPNAMES,
                title=f"Distribution of Step Size During Locomotion ({', '.join(GROUPNAMES)})",
                savedir=ABOVE_FOLDER,
                savename=f"LocomotionStepSizeDistributionPerMouseGroup_{'_'.join(GROUPNAMES)}",
                binsize=BINSIZE,
                show_figure=False,
                save_figure=True
            )
            print("PROCESSING MICE GROUPED TOGETHER DONE!")