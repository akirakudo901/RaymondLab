# Author: Akira Kudo
# Created: 2024/04/01
# Last updated: 2024/06/26

import os
import sys

import numpy as np

# I will learn about proper packaging and arrangement later...
sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID")
sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut")

from BSOID.bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
from BSOID.feature_analysis_and_visualization.analysis.analyze_mouse_gait import analyze_mouse_gait, analyze_mouse_stepsize_per_mousegroup, extract_average_line_between_paw
from BSOID.feature_analysis_and_visualization.visualization.visualize_mouse_gait import filter_nonpawrest_motion, select_N_locomotion_sequences, visualize_mouse_paw_rests_in_locomomotion, visualize_stepsize_in_locomotion_in_multiple_mice, COL_START
from BSOID.label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

from DeepLabCut.dlc_io.utils import read_dlc_csv_file
from DeepLabCut.temper_with_csv_and_hdf.data_filtering.identify_paw_noise import identify_bodypart_noise_by_impossible_speed
from DeepLabCut.temper_with_csv_and_hdf.data_filtering.filter_based_on_boolean_array import filter_based_on_boolean_array
from DeepLabCut.utils_to_be_replaced_oneday import get_mousename

BODYPARTS = [
        'rightforepaw', 'leftforepaw', 
        'righthindpaw', 'lefthindpaw', 
        'snout', 'tailbase'
        ]


def main(label_path : str, dlc_path : str,
         savedir : str, savename : str, locomotion_label : list, 
         length_limits : tuple, bodyparts : list, threshold : float):
    # read csvs
    label, _ = read_BSOID_labeled_features(csv_path=label_path)
    df = read_dlc_csv_file(dlc_path=dlc_path, include_scorer=False)

    # VERY INTRIGUINGLY, IT TURNS OUT THAT THE LABELS I HAVE OBTAINED ARE CUT SHORT
    # COMPARED TO THE DLC DATA. THE CUT SEEMS TO BE EXACTLY 5, BUT I AM NOT SURE WHY 
    # IT IS THIS NUMBER... TRUNCATING THE DLC DATA SHOULD DO THE WORK FOR NOW.
    if False: # TODO FIX
        EXACT_LEEWAY = 5
        if len(label) != (df.shape[0] - EXACT_LEEWAY):
            raise Exception("Length of label and first dimension of data frame doesn't match: " + 
                            f"{len(label)} vs. {df.shape[0]}...")
        # TODO FIX END

    # TRUNCATE THE DLC DATA
    if len(label) < df.shape[0]:
        print(f"Label ({len(label)} long) is shorter than the DLC data ({df.shape[0]} long) " + 
              ", but we know the DLC data should match the labels at the beginning.")
        print("We will hence truncate the DLC data!")
        df = df.loc[:len(label)-1, :]

    # merge all locomotion labels into one category
    for loc_lbl in locomotion_label:
            if loc_lbl != locomotion_label[0]:
                label[label == loc_lbl] = locomotion_label[0]
    # then filter any bout smaller than filter_size    
    filt_label = filter_bouts_smaller_than_N_frames(label=label, n=5)

    noise_df = identify_bodypart_noise_by_impossible_speed(bpt_data=df,
                                                        bodyparts=bodyparts,
                                                        start=0, 
                                                        end=df.shape[0], # number of rows / timestamps
                                                        savedir=None, 
                                                        save_figure=False, 
                                                        show_figure=False)

    # filter based on obtained noise info
    filtered_df = df 
    for bpt in bodyparts:
        bpt_bool_arr = noise_df[(bpt, "loc_wrng")].to_numpy()

        filtered_df = filter_based_on_boolean_array(
            bool_arr=bpt_bool_arr, df=filtered_df, bodyparts=[bpt], filter_mode="linear"
        )

    # make visualizations of the rests in question
    if True:
        visualize_mouse_paw_rests_in_locomomotion(df=filtered_df,
                                            label=filt_label,
                                            bodyparts=bodyparts,
                                            savedir=savedir,
                                            savename=savename,
                                            length_limits=length_limits,
                                            plot_N_runs=float("inf"),
                                            locomotion_label=locomotion_label,
                                            threshold=threshold,
                                            save_figure=True, 
                                            show_figure=False)
        
    # create the yaml files holding step size
    if True:
        savedir = savedir
        savename = "pawRestDistanceOverTime_{}_lenlim{}To{}_thresh{}.yaml".format(
            mousename, length_limits[0], length_limits[1], threshold
            )
        analyze_mouse_gait(df=filtered_df, label=filt_label, bodyparts=bodyparts, 
                        locomotion_label=locomotion_label, length_limits=length_limits,
                        savedir=savedir, savename=savename, save_result=True)
    
    if False:
        def setup_N_locomotions(df, label, N, locomotion_labels, length_limits):
            starts, ends, _ = select_N_locomotion_sequences(
                label=label, N=N, locomotion_labels=locomotion_labels, 
                length_limits=length_limits
                )
            
            # first, get the distance between consecutive steps
            df, avg_df = filter_nonpawrest_motion(
                df=df, label=label, show_nonpaw=True, threshold=threshold, 
                locomotion_labels=locomotion_labels, average_pawrest=True
                )
            
            data = {}

            for start, end in zip(starts, ends):
                sequence = avg_df[(avg_df[COL_START] >= start) & 
                                (avg_df[COL_START] <=   end)]
            
            data[start] = sequence
            return data
            
        
        
        # quantify the distance between consecutive steps
        # when doing so, align the direction with the average of all paw positions, 
        # weighted in the number of right & left paws.
        savedir = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab"
        savename = "average_line_betweeen_paw.txt"

        locomotions = setup_N_locomotions(df=filtered_df, label=label, N=float("inf"), 
                                        locomotion_labels=locomotion_label, length_limits=length_limits)
        for sequence in locomotions.values():
            extract_average_line_between_paw(pawrest_df=sequence, savedir=savedir, savename=savename)

if __name__ == "__main__":

    # shared constants
    THRESHOLD = 0.5 # DETERMINE A GOOD VALUE!
    LENGTH_LIMITS = (60, None)

    if True: # YAC128
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
        
        LOCOMOTION_LABEL = [38]

        SAVEDIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\fig\pawInk\{}"

        # the following are for step size analysis
        ABOVE_STEPSIZE_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\fig\pawInk"

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
        
        WT_GROUPNAME, HD_GROUPNAME = "FVB", "YAC128"

    else: # Q175
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

        SAVEDIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\fig\pawInk\{}"

        # the following are for step size analysis
        ABOVE_STEPSIZE_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\fig\pawInk"

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

        WT_GROUPNAME, HD_GROUPNAME = "B-6", "Q175"

    if False:
        for lbl, dlc in zip(LABEL_CSVS_PATHS, LABEL_DLC_PATHS):
            if get_mousename(lbl) != get_mousename(dlc):
                raise Exception("Provided pair of label & dlc csvs aren't for the same mouse: \n" + 
                                f" - {os.path.basename(lbl)} \n" + 
                                f" - {os.path.basename(dlc)} \n")
            
            print(f"Analyzing mouse: {get_mousename(lbl)}:")
            print(f"- {os.path.basename(lbl)}")
            print(f"- {os.path.basename(dlc)}")

            # setup SAVEDIR & SAVENAME based on the mouse of interest
            mousename = get_mousename(lbl)

            savedir = SAVEDIR.format(mousename)

            if not os.path.exists(savedir):
                os.mkdir(savedir)
            SAVENAME = f"pawRest_{mousename}"

            if True:
                try:
                    main(label_path=lbl, dlc_path=dlc, savedir=savedir, savename=SAVENAME, 
                        locomotion_label=LOCOMOTION_LABEL, length_limits=LENGTH_LIMITS, 
                        bodyparts=BODYPARTS, threshold=THRESHOLD)
                except Exception as e:
                    print(e)
            else:
                main(label_path=lbl, dlc_path=dlc, savedir=savedir, savename=SAVENAME, 
                    locomotion_label=LOCOMOTION_LABEL, length_limits=LENGTH_LIMITS, 
                    bodyparts=BODYPARTS, threshold=THRESHOLD)
    
    # analyze whether the different groups have significant differences 
    if True:
        wtyaml = [os.path.join(ABOVE_STEPSIZE_FOLDER, folder, yaml) 
                  for folder in [os.path.join(ABOVE_STEPSIZE_FOLDER, fol) 
                                 for fol in os.listdir(ABOVE_STEPSIZE_FOLDER) 
                                 if os.path.isdir(os.path.join(ABOVE_STEPSIZE_FOLDER, fol)) and 
                                 fol in WT_MICE]
                  for yaml in os.listdir(os.path.join(ABOVE_STEPSIZE_FOLDER, folder))
                  if yaml.endswith(".yaml")]
        hdyaml = [os.path.join(ABOVE_STEPSIZE_FOLDER, folder, yaml) 
                  for folder in [os.path.join(ABOVE_STEPSIZE_FOLDER, fol) 
                                 for fol in os.listdir(ABOVE_STEPSIZE_FOLDER) 
                                 if os.path.isdir(os.path.join(ABOVE_STEPSIZE_FOLDER, fol)) and 
                                 fol in HD_MICE]
                  for yaml in os.listdir(os.path.join(ABOVE_STEPSIZE_FOLDER, folder))
                  if yaml.endswith(".yaml")]
        
        save_path = os.path.join(os.path.dirname(SAVEDIR), "mouseStepSizeAverageComparison.txt")
        
        analyze_mouse_stepsize_per_mousegroup(
            yamls1=wtyaml, yamls2=hdyaml, groupnames=[WT_GROUPNAME, HD_GROUPNAME],
            bodyparts=["rightforepaw", "leftforepaw", "righthindpaw", "lefthindpaw"],
            # uses unpaired-t if true, mann whitney u if false
            data_is_normal=True, 
            significance=0.05,
            save_result=True,
            save_to=save_path
            )