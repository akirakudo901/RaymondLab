# Author: Akira Kudo
# Created: 2024/04/03
# Last Updated: 2024/08/30

import os

import numpy as np
import pandas as pd
import tqdm

from bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
from feature_analysis_and_visualization.analysis.analyze_time_spent_per_label import analyze_average_behavior_snippet_length_per_label_per_group, analyze_time_spent_per_label_between_groups, compute_time_spent_per_label, compute_time_spent_per_label_per_group_from_csvs
from feature_analysis_and_visualization.utils import get_mousename
from feature_analysis_and_visualization.behavior_groups import BehaviorGrouping
from label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames



def compute_time_spent_per_label_for_list_of_csv_and_save(
        groups : list, group_names : list, savedir : str, savename : str, show_message : bool=False
        ):
    """
    Computes time spent per label for a list of csvs, and saves them into
    the specified csv file.

    :param List[List[str]] groups: List of a list of csvs, one for each group.
    :param List[str] group_names: List of names of groups, same length as 'groups'.
    :param str savedir: Directory to save the info as csv.
    :param str savename: Name of saved csv file.
    :param bool show_message: Whether to show message of the process. Defaults to false.
    """
    if len(groups) != len(group_names): 
        raise Exception("groups has to have the same length as group_names...")
    
    # first identify all unique labels existing in the data
    all_unique_labels = np.array([])
    unique_labels_in_each_group = [np.unique(read_BSOID_labeled_csv(csv_path)[0])
                                   for group in groups
                                   for csv_path in group]
    for u_l in unique_labels_in_each_group:
        all_unique_labels = np.union1d(all_unique_labels, u_l).astype(np.int64)
    
    # then extract the remaining features
    data, mouse_types = {}, []
    # for every group and its name
    for group, name in zip(groups, group_names):
        if show_message:
            print(f"Processing {name}!")
        # extract the labels for the group
        group_labels = [read_BSOID_labeled_csv(csv_path)[0] for csv_path in group]
        mouse_names  = [get_mousename(csv_path) for csv_path in group]
        # for every extracted label
        for label, mousename in tqdm.tqdm(zip(group_labels, mouse_names)):
            # add the mouse type to the mouse_types array
            mouse_types.append(name)
            # compute the statistics
            unique_labels, framecount, percentage = compute_time_spent_per_label(
                        label=label, label_groups=None)
            # create padded versions so that we can log them in the same dataframe
            framecount_padded = np.zeros(len(all_unique_labels))
            # percentage_padded = np.zeros(len(all_unique_labels))
            for label_idx in range(len(unique_labels)):
                lbl, fc, perc = unique_labels[label_idx], framecount[label_idx], percentage[label_idx]
                framecount_padded[all_unique_labels == lbl] = fc
                # percentage_padded[all_unique_labels == lbl] = perc

            data[f'{mousename}'] = framecount_padded.astype(np.int64)
            # data[f'{name}_percentage'] = percentage_padded
    
    df = pd.DataFrame.from_dict(data)
    df.loc[-1] = mouse_types
    df.set_index(pd.Index(all_unique_labels.tolist() + ['mouseType']), inplace=True)
    print(f"Saving of {savename}... ", end="")
    df.to_csv(os.path.join(savedir, savename))
    print(f"SUCCESSFUL!")

def compute_run_lengths_per_label_for_list_of_csv_and_save(
        groups : list, group_names : list, savedir : str, savename : str
        ):
    """
    Computes the length of runs of given labels in the given groups, 
    saving them into a csv.

    :param list groups: _description_
    :param list group_names: _description_
    :param str savedir: _description_
    :param str savename: _description_
    """

if __name__ == "__main__":
    if False:
        YAC_FOLDER = r"C:\Users\mashi\Desktop\temp\YAC"
        WT_FOLDER = r"C:\Users\mashi\Desktop\temp\WT"

        SAVE_DIR = r"C:\Users\mashi\Desktop\temp"

        yac_csvs = [os.path.join(YAC_FOLDER, file) for file in os.listdir(YAC_FOLDER) 
                    if '.csv' in file]
        wt_csvs  = [os.path.join(WT_FOLDER,  file) for file in os.listdir(WT_FOLDER )
                    if '.csv' in file]

        def analyze_and_save_time_spent_per_group():

            all_labels, yac, wt = analyze_time_spent_per_label_between_groups(
                yac_csvs, wt_csvs, label_groups=None)

            framecount_yac, percentage_yac = yac[0], yac[1]
            framecount_wt, percentage_wt = wt[0], wt[1]

            def print_group_framecounts_and_percentages(group_name, framecounts, percentages):
                print(f"{group_name} frames per groups:")
                [print(f"- {lbl}: {count}") for lbl, count in framecounts.items()]
                print(f"{group_name} percentages per groups:")
                [print(f"- {lbl}: {perc}") for lbl, perc in percentages.items()]

            # print_group_framecounts_and_percentages("YAC128", framecount_yac, percentage_yac)
            # print_group_framecounts_and_percentages("WT", framecount_wt, percentage_wt)

            # our_columns = pd.MultiIndex.from_product((['WT', 'YAC128'], ['framecounts', 'percentages']))

            our_columns = pd.Index(['WT framecounts', 'WT percentages', 'YAC128 framecounts', 'YAC128 percentages'])

            index = pd.Index(all_labels, name='BehaviorGroup')
            data = [[framecount_yac[lbl], percentage_yac[lbl], framecount_wt[lbl], percentage_wt[lbl]]
                    for lbl in all_labels]

            df = pd.DataFrame(data, index=index, columns=our_columns)
            df.to_csv(r"C:\Users\mashi\Desktop\temp\labels_spent_over_time.csv")
        
        groups = [yac_csvs, wt_csvs]
        group_names = ["YAC128", "WT"]

        compute_time_spent_per_label_for_list_of_csv_and_save(
            groups=groups, group_names=group_names,
            savedir=SAVE_DIR, savename="time_spent_per_label_per_file_YAC128_WT.csv")

    if False:
        import os
        Q175_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\allcsv_2024_05_16_Akira"
        YAC_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\allcsv_2024_05_16_Akira"
        CSV_FOLDER_LIST = [
            os.path.join(Q175_FOLDER, "HD_filt"),
            os.path.join(Q175_FOLDER, "WT_filt"),
            os.path.join(YAC_FOLDER, "HD_filt"),
            os.path.join(YAC_FOLDER, "WT_filt"),
            ]
        GROUPNAMES = ["Q175", "B6", "YAC128", "FVB"]

        SAVE_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\time_spent_per_label_all_micetype_filt.csv"

        CSV_PATH_LIST = [
            [os.path.join(folder, csv) for csv in os.listdir(folder)] 
             for folder in CSV_FOLDER_LIST
        ]

        compute_time_spent_per_label_per_group_from_csvs(
            csv_groups=CSV_PATH_LIST,
            group_names=GROUPNAMES,
            save_path=SAVE_PATH,
            label_groups=None,
            save_csv=True)
    
    # statistically compare the mean of the frame counts for
    # two groups of labels
    if True:
        # parameters
        SIGNIFICANCE = 0.05
        SAVEDIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\results\framecount_genotype"

        EXCLUDED_MICE = ["308535m1", "312153m2",# HD
                         "326787m2"] #WT

        def is_excluded_mice(path : str, mousenames : list=EXCLUDED_MICE):
            for mousename in mousenames:
                if mousename in path.replace('_', ''):
                    print(f"File {path} is to be excluded as it matches the mouse name: {mousename}.")
                    return True
            return False
        
        def prep_label(csv_path : str, start : int=0, end : int=30*40*60):
            """
            Preps csv by reading and extracting label, then truncating the 
            label to from start to end.
            """
            label, _ = read_BSOID_labeled_features(csv_path)
            return label[start:end+1]

        if True: 
            groupnames = ["YAC128", "FVB"]
            HD_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\\" + \
                r"allcsv_2024_05_16_Akira\HD_filt"
            WT_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\\" + \
                r"allcsv_2024_05_16_Akira\WT_filt"
        else:
            groupnames = ["Q175", "B6"]
            HD_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\\" + \
                r"allcsv_2024_06_20_Akira\HD_filt"
            WT_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\\" + \
                r"allcsv_2024_06_20_Akira\WT_filt"
        
        HD_CSVS = [os.path.join(HD_CSV_DIR, file) 
                   for file in os.listdir(HD_CSV_DIR) 
                   if (file.endswith(".csv") and not is_excluded_mice(file))]
        WT_CSVS = [os.path.join(WT_CSV_DIR, file) 
                   for file in os.listdir(WT_CSV_DIR) 
                   if (file.endswith(".csv") and not is_excluded_mice(file))]
        
        def convert_label_to_int_metalabel(label : np.ndarray, 
                                            behavior_grouping : BehaviorGrouping):
            return np.fromiter([behavior_grouping.label_to_behavioral_group_int(val) 
                                for val in label],
                                dtype=np.int32)

        NETWORK_NAME = 'Feb-23-2023'
        YAML_PATH = r'X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID\feature_analysis_and_visualization\behavior_groups_Akira_Final.yml'
        FILTER_MIN_SIZE = 5
        
        HD_LABELS = [filter_bouts_smaller_than_N_frames(label=prep_label(csv), n=FILTER_MIN_SIZE) 
                        for csv in HD_CSVS]
        WT_LABELS = [filter_bouts_smaller_than_N_frames(label=prep_label(csv), n=FILTER_MIN_SIZE) 
                        for csv in WT_CSVS]

        bg = BehaviorGrouping(network_name=NETWORK_NAME, yaml_path=YAML_PATH)
        groupings = bg.load_behavior_groupings(network_name=NETWORK_NAME, yaml_path=YAML_PATH)

        # convert the labels into integer meta labels
        HD_METALABELS = [convert_label_to_int_metalabel(lbl, bg) for lbl in HD_LABELS]
        WT_METALABELS = [convert_label_to_int_metalabel(lbl, bg) for lbl in WT_LABELS]
        
        # comparison of framecounts of the original labels
        if True:
            SAVENAME = f"framecount_mean_comparison_per_group_per_genotype_{'_'.join(groupnames)}.txt"

            analyze_time_spent_per_label_between_groups(group1=HD_CSVS,
                                                        group2=WT_CSVS,
                                                        groupnames=groupnames, 
                                                        significance=SIGNIFICANCE, 
                                                        savedir=SAVEDIR,
                                                        savename=SAVENAME,
                                                        data_is_normal=False,
                                                        equal_var=False,
                                                        label_groups=None, 
                                                        save_result=True)
        
        # comparison of framecounts of the meta labels
        if True:
            DATA_IS_NORMAL = False
            testtype = "UnpairedT" if DATA_IS_NORMAL else "MWU"

            SAVENAME = f"METALABEL_framecount_mean_comparison_per_group_per_genotype_{'_'.join(groupnames)}_{testtype}_Reviewed2.txt"

            analyze_time_spent_per_label_between_groups(group1=HD_METALABELS,
                                                        group2=WT_METALABELS,
                                                        groupnames=groupnames, 
                                                        significance=SIGNIFICANCE, 
                                                        savedir=SAVEDIR,
                                                        savename=SAVENAME,
                                                        data_is_normal=DATA_IS_NORMAL,
                                                        equal_var=False,
                                                        label_groups=None, 
                                                        save_result=True)
        
        # comparison of mean length of bouts for each label, across genotypes
        # first for original labels
        if True:
            SAVENAME = f"bout_mean_length_comparison_per_genotype_original_label_{'_'.join(groupnames)}.txt"
            DATA_IS_NORMAL = False
            EQUAL_VAR = False

            analyze_average_behavior_snippet_length_per_label_per_group(
                group1=HD_LABELS,
                group2=WT_LABELS,
                groupnames=groupnames,
                significance=SIGNIFICANCE,
                savedir=SAVEDIR,
                savename=SAVENAME,
                data_is_normal=DATA_IS_NORMAL,
                equal_var=EQUAL_VAR,
                label_groups=None, 
                save_result=True
                )
            
        # then for meta labels
        if True:
            SAVENAME = f"bout_mean_length_comparison_per_genotype_meta_label_{'_'.join(groupnames)}.txt"
            DATA_IS_NORMAL = False
            EQUAL_VAR = False

            analyze_average_behavior_snippet_length_per_label_per_group(
                group1=HD_METALABELS,
                group2=WT_METALABELS,
                groupnames=groupnames,
                significance=SIGNIFICANCE,
                savedir=SAVEDIR,
                savename=SAVENAME,
                data_is_normal=DATA_IS_NORMAL,
                equal_var=EQUAL_VAR,
                label_groups=None, 
                save_result=True
                )