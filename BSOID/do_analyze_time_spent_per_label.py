# Author: Akira Kudo
# Created: 2024/04/03
# Last Updated: 2024/05/25

import os

import numpy as np
import pandas as pd
import tqdm

from bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
from feature_analysis_and_visualization.analysis.analyze_time_spent_per_label import analyze_time_spent_per_label_between_groups, compute_time_spent_per_label, compute_time_spent_per_label_per_group_from_csvs
from feature_analysis_and_visualization.utils import get_mousename


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

    if True:
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