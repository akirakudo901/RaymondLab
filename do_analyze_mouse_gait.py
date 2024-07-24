# Author: Akira Kudo
# Created: 2024/04/01
# Last updated: 2024/07/22

import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import levene
import yaml

# I will learn about proper packaging and arrangement later...
sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID")
sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut")

from BSOID.bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
from BSOID.feature_analysis_and_visualization.analysis.analyze_mouse_gait import aggregate_stepsize_per_body_part, analyze_mouse_gait, analyze_mouse_stepsize_per_mousegroup_from_dicts, extract_average_line_between_paw, extract_time_difference_between_consecutive_left_right_contact, filter_stepsize_dict_by_locomotion_to_use, identify_curved_trajectory, remove_outlier_data
from BSOID.feature_analysis_and_visualization.visualization.visualize_mouse_gait import filter_nonpawrest_motion, read_stepsize_yaml, select_N_locomotion_sequences, visualize_mouse_gait_speed_of_specific_sequences, visualize_mouse_paw_rests_in_locomomotion, visualize_stepsize_in_locomotion_in_multiple_mice, visualize_time_between_consecutive_landings, COL_START, STEPSIZE_MIN
from BSOID.label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

from DeepLabCut.dlc_io.utils import read_dlc_csv_file
from DeepLabCut.temper_with_csv_and_hdf.data_filtering.identify_paw_noise import identify_bodypart_noise_by_impossible_speed
from DeepLabCut.temper_with_csv_and_hdf.data_filtering.filter_based_on_boolean_array import filter_based_on_boolean_array
from DeepLabCut.utils_to_be_replaced_oneday import get_mousename
from DeepLabCut.visualization.visualize_qq_plot import visualize_qq_plot_from_dataframe

BODYPARTS = [
        'rightforepaw', 'leftforepaw', 
        'righthindpaw', 'lefthindpaw', 
        'snout', 'tailbase'
        ]
ALL_PAWS = ["rightforepaw", "leftforepaw", "righthindpaw", "lefthindpaw"]


def main(label_path : str, dlc_path : str,
         savedir : str, savename : str, locomotion_label : list, 
         length_limits : tuple, bodyparts : list, threshold : float, 
         sequences : list):
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
    if False:
        visualize_mouse_paw_rests_in_locomomotion(df=filtered_df,
                                            label=filt_label,
                                            bodyparts=bodyparts,
                                            savedir=savedir,
                                            savename=savename,
                                            averaged=True,
                                            annotate_framenum=True,
                                            length_limits=length_limits,
                                            plot_N_runs=float("inf"),
                                            locomotion_label=locomotion_label,
                                            threshold=threshold,
                                            save_figure=True, 
                                            show_figure=False)
        
    # create the yaml files holding step size
    if False:
        savedir = savedir
        mousename = get_mousename(label_path)
        savename = "pawRestDistanceOverTime_{}_lenlim{}To{}_thresh{}.yaml".format(
            mousename, length_limits[0], length_limits[1], threshold
            )
        analyze_mouse_gait(df=filtered_df, label=filt_label, bodyparts=bodyparts, 
                        locomotion_label=locomotion_label, length_limits=length_limits,
                        savedir=savedir, savename=savename, save_result=True)
    
    # taking an already generated yaml file holding step size, 
    # create a yaml that excludes any curved trajectory
    if False:
        def plot_trajectory(X : np.ndarray, Y : np.ndarray, 
                            change_X : np.ndarray, change_Y : np.ndarray,
                            savedir : str,
                            savename : str,
                            zoom_in : bool=False,
                            save_figure : bool=True,
                            show_figure : bool=True):
            _, ax = plt.subplots()
            ax.plot(X, Y, color="red")
            ax.scatter(change_X, change_Y, color="purple")
            if not zoom_in:
                ax.set_xlim(left=0, right=1080)
                ax.set_ylim(bottom=0, top=1080)
            
            if save_figure:
                plt.savefig(os.path.join(savedir, savename))
            if show_figure: plt.show()
            else: plt.close()
        
        SPECIFIC_LEN_LIM = (60, None)
        SPECIFIC_THRESH = 0.5
        
        # also, create subfolders that hold the trajectories that were 
        # excluded and included
        curved, noncurved = os.path.join(savedir, "curved"), os.path.join(savedir, "noncurved")
        if not os.path.exists(curved):    os.mkdir(curved)
        if not os.path.exists(noncurved): os.mkdir(noncurved)

        mousename = get_mousename(label_path)
        savefile = "pawRestDistanceOverTime_{}_lenlim{}To{}_thresh{}.yaml".format(
            mousename, SPECIFIC_LEN_LIM[0], SPECIFIC_LEN_LIM[1], SPECIFIC_THRESH
            )
        with open(os.path.join(savedir, savefile), 'r') as f:
            content = yaml.safe_load(f.read())
        
        non_curved_traj = {}
        
        for start_idx, val in content.items():
            end_idx = val['end']
            X = filtered_df.loc[start_idx:end_idx, ('tailbase', 'x')].to_numpy()
            Y = filtered_df.loc[start_idx:end_idx, ('tailbase', 'y')].to_numpy()
            change_at = identify_curved_trajectory(X=X, Y=Y, windowsize=20, 
                                                   threshold=np.pi/4)
            if len(change_at) == 0:
                non_curved_traj[start_idx] = val
            
            savedir = noncurved if len(change_at) == 0 else curved
            savename = "tailbaseTrajWithTurnPoints_{}_{}To{}_lenlim{}To{}_thresh{}.png".format(
                mousename, start_idx, end_idx, SPECIFIC_LEN_LIM[0], SPECIFIC_LEN_LIM[1], SPECIFIC_THRESH
                )
            plot_trajectory(
                X=X, Y=Y, change_X=X[change_at], change_Y=Y[change_at], 
                zoom_in=False, savedir=savedir, savename=savename, 
                save_figure=True, show_figure=False)
            
        
        with open(os.path.join(savedir, 
                               savefile.replace('.yaml', '_noncurved.yaml')), 
                               'w') as f:
            to_write = yaml.dump(non_curved_traj)
            f.write(to_write)    
    
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

    
    if False:
        if sequences is None:
            print("No sequence to analyze for this mouse...")
        else:
            savename = "ConsecutiveRightLeftContact.yaml"

            extract_time_difference_between_consecutive_left_right_contact(
                df=df, 
                label=label, 
                sequences=sequences, 
                savedir=savedir, savename=savename, 
                comparison_pairs=[
                                ["rightforepaw", "leftforepaw"],
                                ["righthindpaw", "lefthindpaw"],
                                ],
                ignore_close_paw=STEPSIZE_MIN,
                locomotion_label=LOCOMOTION_LABEL,
                save_result=True
            )

    # visualize mouse gait for specific sequences
    if False:
        if sequences is None:
            print("No sequence to analyze for this mouse...")
        else:
            visualize_mouse_gait_speed_of_specific_sequences(
                df=df,
                sequences=sequences,
                bodyparts=['rightforepaw', 'leftforepaw', 'righthindpaw', 'lefthindpaw'],
                paw_rest_color="orange",
                savedir=savedir,
                save_prefix="ThresholdPawRest",
                threshold=THRESHOLD,
                save_figure=True,
                show_figure=False
            )

if __name__ == "__main__":

    # shared constants
    THRESHOLD = 0.75 # DETERMINE A GOOD VALUE!
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

        STEPS_TO_USE = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\YAC128_locomotion_to_use.yaml"

        # Yamls for time difference between consecutive body part steps
        ABOVE_TIME_DIFFERENCE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\fig\pawInk"
        TIME_DIFFERENCE_FILENAME = "ConsecutiveRightLeftContact.yaml"
        STEP_TIME_DIFFERENCE_YAMLS = [os.path.join(ABOVE_TIME_DIFFERENCE_DIR, d, TIME_DIFFERENCE_FILENAME)
                                      for d in os.listdir(ABOVE_TIME_DIFFERENCE_DIR) 
                                      if (os.path.isdir(os.path.join(ABOVE_TIME_DIFFERENCE_DIR, d)) and 
                                          os.path.exists(os.path.join(ABOVE_TIME_DIFFERENCE_DIR, d, TIME_DIFFERENCE_FILENAME)))]

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

    if True:
        # identify which locomotion sequence to use based on STEPS_TO_USE
        with open(STEPS_TO_USE, 'r') as f:
            content = f.read()
            locomotion_to_use = yaml.safe_load(content)

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

            # data in order to work with consecutive paw landings
            sequences_to_use_for_this_mouse = locomotion_to_use[mousename]

            # do a bunch of stuff
            if False:
                try:
                    main(label_path=lbl, dlc_path=dlc, savedir=savedir, savename=SAVENAME, 
                        locomotion_label=LOCOMOTION_LABEL, length_limits=LENGTH_LIMITS, 
                        bodyparts=BODYPARTS, threshold=THRESHOLD, 
                        sequences=sequences_to_use_for_this_mouse)
                except Exception as e:
                    print(e)
            else:
                main(label_path=lbl, dlc_path=dlc, savedir=savedir, savename=SAVENAME, 
                    locomotion_label=LOCOMOTION_LABEL, length_limits=LENGTH_LIMITS, 
                    bodyparts=BODYPARTS, threshold=THRESHOLD, 
                    sequences=sequences_to_use_for_this_mouse)
    
    # analyze whether the different groups have significant differences 
    if False:
        # obtain all yamls of interest
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
        
        # filter based on STEPS_TO_USE
        with open(STEPS_TO_USE, 'r') as f:
            content = f.read()
            locomotion_to_use = yaml.safe_load(content)

        wtdicts = [filter_stepsize_dict_by_locomotion_to_use(
                    dict=read_stepsize_yaml(yaml), 
                    mousename=get_mousename(yaml),
                    locomotion_to_use=locomotion_to_use) 
                for yaml in wtyaml]
        hddicts = [filter_stepsize_dict_by_locomotion_to_use(
                    dict=read_stepsize_yaml(yaml), 
                    mousename=get_mousename(yaml),
                    locomotion_to_use=locomotion_to_use) 
                for yaml in hdyaml]
        
        # check for normality with q-q plots / for equality of variances
        if True:
            SELECTED_SAVEDIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\fig\pawInk\selected"
            
            wt_stepsizes = aggregate_stepsize_per_body_part(dictionaries=wtdicts,
                                                            bodyparts=ALL_PAWS, 
                                                            cutoff=STEPSIZE_MIN)
            hd_stepsizes = aggregate_stepsize_per_body_part(dictionaries=hddicts, 
                                                            bodyparts=ALL_PAWS,
                                                            cutoff=STEPSIZE_MIN)
            # pad the entries of stepsizes so as to make a data frame
            def remove_outlier(stepsizes : dict):
                for key, val in stepsizes.items():
                    stepsizes[key] = remove_outlier_data(np.array(val))
                return stepsizes
            
            def pad_stepsizes(stepsizes : dict):
                maxlen = max([len(ss) for ss in stepsizes.values()])
                for key, val in stepsizes.items():
                    pad = np.empty(maxlen - len(val)); pad.fill(np.nan)
                    stepsizes[key] = np.concatenate((val, pad))
                return stepsizes
                
            wt_no_outlier = remove_outlier(wt_stepsizes)
            hd_no_outlier = remove_outlier(hd_stepsizes)
            
            # wt_padded_no_outlier = pad_stepsizes(wt_no_outlier)
            # hd_padded_no_outlier = pad_stepsizes(hd_no_outlier)
            
            # Q-Q plot
            if False:
                # make the data frame
                for mousetype, stepsizes in zip(["YAC128", "FVB"], 
                                                [hd_padded_no_outlier, wt_padded_no_outlier]):
                    df = pd.DataFrame(data=stepsizes)
                    visualize_qq_plot_from_dataframe(df=df,
                                                    columns=ALL_PAWS,
                                                    save_dir=SELECTED_SAVEDIR,
                                                    save_prefix="Selected",
                                                    mousetype=mousetype,
                                                    dist=None,
                                                    dist_name="Standard Normal",
                                                    line=None, 
                                                    fit=False,
                                                    save_figure=True,
                                                    show_figure=False)
            
            # compare the variances
            if False:
                ALPHA = 0.05
                for bpt in ALL_PAWS:
                    
                    statistic, pvalue = levene(wt_no_outlier[bpt], hd_no_outlier[bpt], 
                                               center='median')
                    print(f"P-value for Brown-Forsythe test for {bpt}:")
                    print(f"- {pvalue} {'<' if pvalue < ALPHA else '>'} {ALPHA}!")


        # compare the mean
        if True:
            save_path = os.path.join(os.path.dirname(SAVEDIR), 
                                    "Selected_Welch_mouseStepSizeAverageComparison.txt")
            
            analyze_mouse_stepsize_per_mousegroup_from_dicts(
                dicts1=wtdicts, dicts2=hddicts, groupnames=[WT_GROUPNAME, HD_GROUPNAME],
                bodyparts=ALL_PAWS,
                # uses unpaired-t if true, mann whitney u if false
                data_is_normal=False, 
                significance=0.05,
                save_result=True,
                save_to=save_path
                )
            
    # visualize content of YAMLs containing time difference between consecutive landing
    # of specific body pairs (e.g. left / right paws) 
    if False:
        visualize_time_between_consecutive_landings(
            yamls=STEP_TIME_DIFFERENCE_YAMLS, 
            savedir=os.path.dirname(SAVEDIR),
            save_prefix="firstTry",
            binsize=1,
            save_figure=True,
            show_figure=False
            )