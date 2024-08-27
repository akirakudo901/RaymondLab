# Author: Akira Kudo
# Created: 2024/05/17
# Last Updated: 2024/08/09

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from bsoid_io.utils import read_BSOID_labeled_features
from feature_analysis_and_visualization.analysis.analyze_mouse_gait import aggregate_stepsize_per_body_part, aggregate_fore_hind_paw_as_one, filter_stepsize_dict_by_locomotion_to_use, remove_outlier_data, FOREPAW, HINDPAW
from feature_analysis_and_visualization.utils import get_mousename
from feature_analysis_and_visualization.visualization.visualize_mouse_gait import read_stepsize_yaml, visualize_locomotion_stats, visualize_mouse_gait_speed, visualize_stepsize_standard_deviation_per_mousegroup, visualize_stepsize_in_locomotion, visualize_stepsize_in_locomotion_in_multiple_mice, visualize_stepsize_in_locomotion_in_single_mouse, visualize_stepsize_in_locomotion_in_mice_groups, STEPSIZE_MIN
from label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

BINSIZE = 10#20
ALL_PAWS = ['rightforepaw', 'leftforepaw', 'righthindpaw', 'lefthindpaw']

# YAC128
if True:
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
    SAVEDIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\fig\pawInk\fig_workTermReport"

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
    
    STEPS_TO_USE = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\YAC128_locomotion_to_use.yaml"

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
    with open(STEPS_TO_USE, 'r') as f:
        content = f.read()
        locomotion_to_use = yaml.safe_load(content)

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

                    stepsize_info = read_stepsize_yaml(even_fullpath)
                    stepsize_info = filter_stepsize_dict_by_locomotion_to_use(
                        dict=stepsize_info,
                        mousename=mousename,
                        locomotion_to_use=locomotion_to_use
                        )
                    
                    visualize_stepsize_in_locomotion_in_single_mouse(
                        stepsize_info=stepsize_info,
                        title=f"Distance Between Consecutive Rests ({mousename})",
                        savedir=fullpath,
                        savename=f"selected_distanceBetweenConsecutiveRests_{mousename}",
                        binsize=BINSIZE,
                        show_figure=False,
                        save_figure=True
                    )
                    break
        print("SINGLE MICE DONE!")

    if True:
        def filter_yamls_by_mousenames_and_stepsizes(micename : list,
                                                     locomotion_to_use : dict):
            filt_by_micename =  [os.path.join(ABOVE_FOLDER, folder, yaml) 
                                 for folder in [os.path.join(ABOVE_FOLDER, fol) 
                                                for fol in os.listdir(ABOVE_FOLDER) 
                                                if os.path.isdir(os.path.join(ABOVE_FOLDER, fol)) and 
                                                fol in micename]
                                for yaml in os.listdir(os.path.join(ABOVE_FOLDER, folder))
                                if (yaml.startswith("pawRestDistanceOverTime") and 
                                    yaml.endswith(".yaml"))]
            
            filt_to_use = [filter_stepsize_dict_by_locomotion_to_use(
                dict=read_stepsize_yaml(yaml), 
                mousename=get_mousename(yaml),
                locomotion_to_use=locomotion_to_use) 
                for yaml in filt_by_micename
                ]
            matching_micename = [get_mousename(yaml) for yaml in filt_by_micename]
            return filt_to_use, matching_micename

        YAMLS = [os.path.join(ABOVE_FOLDER, folder, yaml) 
                for folder in [os.path.join(ABOVE_FOLDER, fol) 
                               for fol in os.listdir(ABOVE_FOLDER) 
                               if os.path.isdir(os.path.join(ABOVE_FOLDER, fol))]
                for yaml in os.listdir(os.path.join(ABOVE_FOLDER, folder))
                if (yaml.startswith("pawRestDistanceOverTime") and 
                    yaml.endswith(".yaml"))]
        
        all_dicts = [filter_stepsize_dict_by_locomotion_to_use(
                    dict=read_stepsize_yaml(yaml), 
                    mousename=get_mousename(yaml),
                    locomotion_to_use=locomotion_to_use) 
                for yaml in YAMLS]
        all_micename = [get_mousename(path) for path in YAMLS]
        
        wt_dicts, wt_micename = filter_yamls_by_mousenames_and_stepsizes(WT_MICE, locomotion_to_use)
        hd_dicts, hd_micename = filter_yamls_by_mousenames_and_stepsizes(HD_MICE, locomotion_to_use)
        male_dicts, male_micename     = filter_yamls_by_mousenames_and_stepsizes(MALE_MICE, locomotion_to_use)
        female_dicts, female_micename = filter_yamls_by_mousenames_and_stepsizes(FEMALE_MICE, locomotion_to_use)

        if False:
            DICTGROUPS = [all_dicts, hd_dicts, wt_dicts, male_dicts, female_dicts]
            MICENAME_GROUPS = [all_micename, hd_micename, wt_micename, male_micename, female_micename]
            if "Q175" in ABOVE_FOLDER:
                GROUPNAMES = ["Q175-B6", "Q175", "B6", "Male Q175-B6", "Female Q175-B6"]
            elif "YAC128" in ABOVE_FOLDER:
                GROUPNAMES = ["YAC128-FVB", "YAC128", "FVB", "Male YAC128-FVB", "Female YAC128-FVB"]

            print("Processing mice individually per group!")
            for dict_group, groupname, mousenames in zip(DICTGROUPS, GROUPNAMES, MICENAME_GROUPS):
                print(f"Processing group: {groupname}!")
                visualize_stepsize_in_locomotion_in_multiple_mice(
                    stepsizes=dict_group,
                    mousenames=mousenames,
                    title=f"Frequency of Step Size Locomotion In {groupname} Mice",
                    savedir=ABOVE_FOLDER,
                    savename=f"selected_VisualizeStepsizeInLocomotionIn{groupname.replace(' ', '')}",
                    binsize=BINSIZE,
                    show_figure=False,
                    save_figure=True
                )
            
            print("PROCESSING MICE INDIVIDUALLY PER GROUP DONE!")
        
        if True:
            REMOVE_OUTLIERS = True

            # YAC128
            if "YAC128" in ABOVE_FOLDER:
                if True:
                    GROUPNAMES = ["YAC128", "FVB"]
                    GROUPCOLORS = ["pink", "grey"]
                    DICTGROUPS = [hd_dicts, wt_dicts]
                    
                else:
                    GROUPNAMES = ["Male YAC128-FVB", "Female YAC128-FVB"]
                    GROUPCOLORS = ["cyan", "pink"]
                    DICTGROUPS = [male_dicts, wt_dicts]
            
            # Q175
            elif "Q175" in ABOVE_FOLDER:
                if True:
                    GROUPNAMES = ["Q175", "B6"]
                    GROUPCOLORS = ["red", "blue"]
                    DICTGROUPS = [hd_dicts, wt_dicts]
                else:
                    GROUPNAMES = ["Male Q175-B6", "Female Q175-B6"]
                    GROUPCOLORS = ["cyan", "pink"]
                    DICTGROUPS = [male_dicts, wt_dicts]
            

            if False:
                print("Processing mice grouped together!")
                
                if REMOVE_OUTLIERS:
                    # remove any outlier before visualizing
                    no_outlier_dicts = []
                    for dicts in DICTGROUPS:
                        merged_dict = aggregate_stepsize_per_body_part(
                            dictionaries=dicts, bodyparts=ALL_PAWS, cutoff=STEPSIZE_MIN)
                        for key, val in merged_dict.items():
                            merged_dict[key] = {"diff" : remove_outlier_data(np.array(val))}
                        merged_dict["end"] = None
                        no_outlier_dicts.append([{"dummy" : merged_dict}]) # dummy level to make compatible
                        # with visualizing code that follows

                if REMOVE_OUTLIERS:
                    title = f"Distribution of Step Size During Locomotion \nNo Outliers ({', '.join(GROUPNAMES)}) - Binsize={BINSIZE}"
                    savename = f"selected_NoOutlier_LocomotionStepSizeDistributionPerMouseGroup_{'_'.join(GROUPNAMES)}"
                else:
                    title = f"Distribution of Step Size During Locomotion \nWith Outliers ({', '.join(GROUPNAMES)}) - Binsize={BINSIZE}"
                    savename = f"selected_WithOutlier_LocomotionStepSizeDistributionPerMouseGroup_{'_'.join(GROUPNAMES)}"

                visualize_stepsize_in_locomotion_in_mice_groups(
                    stepsize_groups=no_outlier_dicts,
                    groupnames=GROUPNAMES,
                    title=title,
                    savedir=SAVEDIR,
                    savename=savename,
                    group_colors=GROUPCOLORS,
                    binsize=BINSIZE,
                    show_figure=False,
                    save_figure=True
                )
                print("PROCESSING MICE GROUPED TOGETHER DONE!")

            # visualize the standard deviation of step sizes
            if True:
                ALSO_ANALYZE_FORE_HINDPAWS_AGGREGATED = True
                BODYPARTS = ALL_PAWS
                # parameters for SD figure making
                SIGNIFICANCE = 0.05
                COLORS = ["pink", "black"]
                DATA_IS_NORMAL = False
                SHOW_MEAN = True
                SHOW_MOUSENAME = False
                SAVE_FIGURE = False
                SHOW_FIGURE = False
                SAVE_MEAN_COMPARISON_RESULT = False
                
                print("Processing SD of step sizes per group!")
                if REMOVE_OUTLIERS:
                    # remove any outlier before visualizing
                    dictgroups_no_outlier = []
                    for dicts in DICTGROUPS:
                        no_outlier_dicts = []
                        for d in dicts:
                            if len(d) == 0: 
                                merged_dict = {}
                                for bpt in BODYPARTS:
                                    merged_dict[bpt] = {"diff" : np.array([])}
                                merged_dict['end'] = None
                            else:
                                merged_dict = aggregate_stepsize_per_body_part(
                                    dictionaries=[d], bodyparts=ALL_PAWS, cutoff=STEPSIZE_MIN)
                                for key, val in merged_dict.items():
                                    merged_dict[key] = {"diff" : remove_outlier_data(np.array(val))}
                                merged_dict['end'] = None
                                
                            no_outlier_dicts.append({"dummy" : merged_dict}) # dummy level to make compatible
                            # with visualizing code that follows
                        dictgroups_no_outlier.append(no_outlier_dicts)

                
                savename = f"selected_NoOutlier_LocomotionStepSizeStandardDevPerMouseGroup_{'_'.join(GROUPNAMES)}"

                visualize_stepsize_standard_deviation_per_mousegroup(
                    stepsize_groups=dictgroups_no_outlier,
                    mousenames=[hd_micename, wt_micename],
                    groupnames=GROUPNAMES, data_is_normal=DATA_IS_NORMAL,
                    significance=SIGNIFICANCE, colors=COLORS, bodyparts=BODYPARTS, 
                    savedir=ABOVE_FOLDER,
                    savename=savename,
                    # cutoff : float=STEPSIZE_MIN,
                    show_mean=SHOW_MEAN,
                    show_mousename=SHOW_MOUSENAME,
                    save_figure=SAVE_FIGURE,
                    show_figure=SHOW_FIGURE,
                    save_mean_comparison_result=SAVE_MEAN_COMPARISON_RESULT
                    )
                
                if ALSO_ANALYZE_FORE_HINDPAWS_AGGREGATED:
                    aggregated_dict_groups = [[aggregate_fore_hind_paw_as_one(d) 
                                              for d in dictgroup]
                                              for dictgroup in dictgroups_no_outlier]
                    visualize_stepsize_standard_deviation_per_mousegroup(
                        stepsize_groups=aggregated_dict_groups,
                        mousenames=[hd_micename, wt_micename],
                        groupnames=GROUPNAMES, data_is_normal=DATA_IS_NORMAL,
                        significance=SIGNIFICANCE, colors=COLORS, 
                        bodyparts=[FOREPAW, HINDPAW], 
                        savedir=ABOVE_FOLDER,
                        savename=savename,
                        # cutoff : float=STEPSIZE_MIN,
                        show_mean=SHOW_MEAN,
                        show_mousename=SHOW_MOUSENAME,
                        save_figure=SAVE_FIGURE,
                        show_figure=SHOW_FIGURE,
                        save_mean_comparison_result=SAVE_MEAN_COMPARISON_RESULT
                        )
                
                print("PROCESSING STEP SIZE PER MOUSE GROUP DONE!")
    
        if True:
            all_micename