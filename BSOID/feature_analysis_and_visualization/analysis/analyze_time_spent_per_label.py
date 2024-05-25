# Author: Akira Kudo
# Created: 2024/04/02
# Last Updated: 2024/05/25

from typing import List

import numpy as np
import pandas as pd
import tqdm

from bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
from feature_analysis_and_visualization.utils import get_mousename

def compute_time_spent_per_label(label : np.ndarray, 
                                 label_groups : List[int]=None):
    """
    Analyzes the number of frames-spent-per-behavior-label in a given
    label numpy array, and returns the result in: 
    - list of unique labels we examined
    - number of frames spent in each groups in the same order
    - percentage of time spent in each groups in the same order

    :param np.ndarray label: Label extracted from a csv.
    :param List[int] label_groups: A list of labels (integers) specifying
    which groups to consider for analysis. Defaults to None - every label.

    :returns np.ndarray unique_labels: List of unique labels we examined.
    :returns np.ndarray raw_frame_counts: Number of frames spent in each groups in the same order.
    :returns np.ndarray percentage: Percentage of time spent in each groups in the same order.
    """
    unique_labels = np.sort(np.unique(label)).astype(np.int64)
    # depending on groups_to_check, filter which group label we consider
    if label_groups is not None and len(label_groups) != 0:
        keep_these_indices = np.isin(unique_labels, np.array(label_groups))
        unique_labels = unique_labels[keep_these_indices]
        unique_labels.sort()

    if len(unique_labels) == 0: 
        return np.array([]), np.array([]), np.array([])
    
    # compute the values of interest
    raw_frame_counts = []
    for unique_lbl in unique_labels:
        count = np.sum(label == unique_lbl)
        raw_frame_counts.append(count)

    raw_frame_counts = np.array(raw_frame_counts).astype(np.int64)
    percentage = raw_frame_counts / np.sum(raw_frame_counts) * 100
    
    return unique_labels, raw_frame_counts, percentage

def compute_time_spent_per_label_per_group_from_csvs(
        csv_groups : List[List[str]],
        group_names : List[str],
        save_path : str,
        label_groups : List[int]=None,
        save_csv : bool=True,
        show_message : bool=False):
    """
    Computes the time spent per label in a given DLC csv, for each group 
    of csvs passed. Then save the result into a csv, indicating:
    - the number of frames spent by a given mouse in a given label group
    - the framecount converted into a percentage spent in a given label group
    - the name of the 'group' the mouse belongs to (useful for marking genotype)

    :param List[List[str]] csv_groups: List of 'groups of csvs', each being a 
    distinct 'group' of csv paths that hold BSOID mouse data.
    :param List[str] group_names: List of the name of the corresponding groups.
    :param str save_path: Path to which we save the csv, including the name of the csv.
    :param List[int] label_groups: List of integer indicating which label groups
    to consider when storing their time spent, defaults to all labels.
    :param bool save_csv: Whether to save the csv, defaults to true.
    :param bool show_message: Whether to show message of the process. Defaults to false.
    """
    nparray_groups = [[read_BSOID_labeled_csv(csv)[0] for csv in group] for group in csv_groups]
    mousenames = [[get_mousename(csv) for csv in group] for group in csv_groups]
    df = compute_time_spent_per_label_per_group_from_numpy_array(
        nparray_groups=nparray_groups,
        mousenames=mousenames,
        group_names=group_names,
        save_path=save_path,
        label_groups=label_groups,
        save_csv=save_csv,
        show_message=show_message)
    return df

def compute_time_spent_per_label_per_group_from_numpy_array(
        nparray_groups : List[List[np.ndarray]],
        mousenames : List[List[str]],
        group_names : List[str],
        save_path : str,
        label_groups : List[int]=None,
        save_csv : bool=True, 
        show_message : bool=True):
    """
    Computes the time spent per label in a given DLC label np.array, for each group 
    of np.array passed. Then save the result into a csv, indicating:
    - the number of frames spent by a given mouse in a given label group
    - the framecount converted into a percentage spent in a given label group
    - the name of the 'group' the mouse belongs to (useful for marking genotype)

    :param List[List[str]] nparray_groups: List of 'groups of numpy array', each 
    being a distinct 'group' of np.ndarray that hold mouse openfield label data.
    :param List[List[str]] mousenames: List of groups of names of mice, each
    corresponding to the numpy array contained in nparray_groups. Expected to be
    the same shape as nparray_groups.
    :param List[str] group_names: List of the name of the corresponding groups.
    :param str save_path: Path to which we save the csv, including the name of the csv.
    :param List[int] label_groups: List of integer indicating which label groups
    to consider when storing their time spent, defaults to all labels.
    :param bool save_csv: Whether to save the csv, defaults to true.
    :param bool show_message: Whether to show message of the process. Defaults to false.
    """
    if len(nparray_groups) == 0: return
    if len(nparray_groups) != len(group_names):
        raise Exception("Length of nparray_groups and group_names must be the same, but " + 
                        f"were instead: {len(nparray_groups)} and {len(group_names)}...")
    if len(nparray_groups) != len(mousenames) or \
        any([len(nparr) != len(mouse) for nparr, mouse in zip(nparray_groups, mousenames)]):
        raise Exception("The overall shape of nparray_groups and mousenames must be the same, " + 
                        "but seem to differ...")
    
    data_per_mouse, data_for_df = {}, []
    # extract the unique labels and their occurrence data first
    for group, grpname, grp_mousenames in zip(nparray_groups, group_names, mousenames):
        if show_message:
            print(f"Processing group: {grpname}!")
        for nparr, mousename in tqdm.tqdm(zip(group, grp_mousenames)):
            unique_labels, raw_frame_counts, percentage = compute_time_spent_per_label(
                label=nparr, label_groups=label_groups
            )
            data_per_mouse[mousename] = (unique_labels, raw_frame_counts, percentage, grpname)
    # then put it together into a pd.DataFrame to save as csv
    # first identify all the unique labels
    all_unique_labels = [val[0] for val in data_per_mouse.values()]
    all_unique_labels = np.unique(np.concatenate(all_unique_labels))
    # construct rows with the format:
    # mousename | group1_framecount | group2_count | ... | groupN_count | groupname
    # mousename | group1_percentage | group2_perct | ... | groupN_perct | groupname
    for mousename in data_per_mouse.keys():
        unique_labels, raw_frame_counts, percentage, grpname = data_per_mouse[mousename]
        # get the sorted framecount & percentage entries into all-zero arrays
        indices = np.isin(element=all_unique_labels, test_elements=unique_labels)
        framecount_in_row, percentage_in_row = np.zeros_like(all_unique_labels), np.zeros_like(all_unique_labels)
        framecount_in_row[indices] = raw_frame_counts
        percentage_in_row[indices] = percentage
        # form the rows
        framecount_row = [mousename] + framecount_in_row.tolist() + [grpname]
        percentage_row = [mousename] + percentage_in_row.tolist() + [grpname]
        # insert the rows into the data
        data_for_df.append(framecount_row); data_for_df.append(percentage_row)

    # create the dataframe
    df = pd.DataFrame(data=data_for_df, 
                      columns=["mousename"]+[f"group{lbl}" for lbl in all_unique_labels] + \
                              ["groupname"],
                      index=pd.MultiIndex.from_product((list(data_per_mouse.keys()), 
                                                        ['framecount', 'percentage'])))
    if save_csv:
        df.to_csv(save_path)
    return df

def analyze_time_spent_per_label_between_groups(
        group1 : List[str],
        group2 : List[str],
        label_groups : List[int]=None,
        ):
    """
    Run analysis on aggregated data of groups, passed as lists to 
    the csvs of interest.
    
    Will examine:
    - The difference in average time spent between the same behavior 
      label among the group.
      <- unpaired t-test?

    :param List[str] group1: The first group as list of csv paths.
    :param List[str] group2: The second group as list of csv paths.
    :param List[int] label_groups: A list of all group labels to 
    examine for comparion. Defaults to None - all labels.
    :returns List[int] all_labels: All unique labels, sorted ascending.
    :returns (total_framcounts1, avg_percentages1): Both dictionaries 
    mapping the label to its corresponding value.
    :returns (total_framcounts2, avg_percentages2): Same for group 2.
    """
    def compute_statsistics_for_group(label_array : list):
        """Takes a list of label np.ndarrays."""
        framecounts, percentages = {},{}
        for lbl in label_array:
            unique_labels, framecount, percentage = compute_time_spent_per_label(
                label=lbl, label_groups=label_groups)
            for idx, unique_l in enumerate(unique_labels):
                fc, perc = framecount[idx].item(), percentage[idx].item()
                framecounts[unique_l] = framecounts.get(unique_l, 0) + fc
                percentages[unique_l] = percentages.get(unique_l, 0) + perc
        # resort the label orders
        sorted_keys = list(framecounts.keys()); sorted_keys.sort()
        srtd_framecounts, srtd_percentages = {},{}
        for k in sorted_keys:
            srtd_framecounts[k] = framecounts[k]
            srtd_percentages[k] = percentages[k]
        return srtd_framecounts, srtd_percentages

    group1_labels = [read_BSOID_labeled_csv(csv)[0] for csv in group1]
    group2_labels = [read_BSOID_labeled_csv(csv)[0] for csv in group2]
    # compute statistics
    total_framecounts1, total_percentages1 = compute_statsistics_for_group(group1_labels)
    total_framecounts2, total_percentages2 = compute_statsistics_for_group(group2_labels)
    # compute average percentages
    avg_percentages1, avg_percentages2 = {},{}
    for k, v in total_percentages1.items(): avg_percentages1[k] = v / len(group1_labels)
    for k, v in total_percentages2.items(): avg_percentages2[k] = v / len(group2_labels)
    # find the union of labels existing in all data
    all_labels = list(total_framecounts1.keys())
    [all_labels.append(k) for k in total_framecounts2.keys() if k not in all_labels]
    all_labels.sort()
    
    return all_labels, (total_framecounts1, avg_percentages1), (total_framecounts2, avg_percentages2)
    


if __name__ == "__main__":
    if False:
        # sanity check: compute_time_spent_per_label
        label = np.array([0]*25 + [1]*32 + [2]*34 + [0]*100)
        groups = [[0,1,2],
                [0,2], 
                [],
                [1],
                [2,1,0]]

        for g in groups:
            u_groups, framecounts, percentage = compute_time_spent_per_label(label=label, label_groups=g)
            print(f"g: {g}, u_groups: {u_groups}")
            print(f"framecounts: {framecounts}, percentage: {percentage}")
    
    if True:
        import os
        Q175_FOLDER = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\allcsv_2024_05_16_Akira"
        YAC_FOLDER = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\allcsv_2024_05_16_Akira"
        CSV_FOLDER_LIST = [
            os.path.join(Q175_FOLDER, "HD_filt"),
            os.path.join(Q175_FOLDER, "WT_filt"),
            os.path.join(YAC_FOLDER, "HD_filt"),
            os.path.join(YAC_FOLDER, "WT_filt"),
            ]
        GROUPNAMES = ["Q175", "B6", "YAC128", "FVB"]

        CSV_PATH_LIST = [
            [os.path.join(folder, csv) for csv in os.listdir(folder)] for folder in CSV_FOLDER_LIST
        ]

        compute_time_spent_per_label_per_group_from_csvs(
            csv_groups=CSV_PATH_LIST,
            group_names=GROUPNAMES,
            save_path=None,
            label_groups=None,
            save_csv=False)