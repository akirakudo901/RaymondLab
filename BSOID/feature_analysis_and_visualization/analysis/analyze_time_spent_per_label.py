# Author: Akira Kudo
# Created: 2024/04/02
# Last Updated: 2024/04/02

from typing import List

import numpy as np

from bsoid_io.utils import read_BSOID_labeled_csv

def compute_time_spent_per_label(label : np.ndarray, 
                                 groups_to_check : List[int]=None
                                 ):
    """
    Analyzes the number of frames-spent-per-behavior-label in a given
    label numpy array, and returns the result in: 
    - original frames spent in each groups
    - percentage of frames spent in each groups

    :param np.ndarray label: Label extracted from a csv.
    :param List[int] groups_to_check: A list of labels (integers) specifying
    which groups to consider for analysis. Defaults to None - every label.
    """
    unique_groups = np.sort(np.unique(label))
    # depending on groups_to_check, filter which group label we consider
    if groups_to_check is not None and len(groups_to_check) != 0:
        keep_these_indices = np.isin(unique_groups, np.array(groups_to_check))
        unique_groups = unique_groups[keep_these_indices]
    # compute the values of interest
    raw_frame_counts = []
    for unique_lbl in unique_groups:
        count = np.sum(label == unique_lbl)
        raw_frame_counts.append(count)

    raw_frame_counts = np.array(raw_frame_counts)
    percentage = raw_frame_counts / np.sum(raw_frame_counts) * 100
    
    return unique_groups, raw_frame_counts, percentage


def analyze_time_spent_per_label_between_groups(
        group1 : List[str],
        group2 : List[str],
        groups_to_check : List[int]=None
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
    :param List[int] groups_to_check: A list of all group labels to 
    examine for comparion. Defaults to None - all labels.
    """
    group1_labels = [read_BSOID_labeled_csv(csv)[0] for csv in group1]
    group2_labels = [read_BSOID_labeled_csv(csv)[0] for csv in group2]
    # compute time spent per label
    list_of_time_spent_per_label_g1 = [compute_time_spent_per_label(
        label=lbl, groups_to_check=groups_to_check)
        for lbl in group1_labels]
    list_of_time_spent_per_label_g2 = [compute_time_spent_per_label(
        label=lbl, groups_to_check=groups_to_check)
        for lbl in group2_labels]
    # find the union of labels existing in all data
    all_labels = np.unique(np.array(
        [t[0] for t in list_of_time_spent_per_label_g1] + 
        [t[0] for t in list_of_time_spent_per_label_g2]
        ))
    # also compute the sum of raw frame counts per group
    frame_count_group1 = np.sum(np.array(
        [t[1] for t in list_of_time_spent_per_label_g1]
    ), axis=0)
    frame_count_group2 = np.sum(np.array(
        [t[1] for t in list_of_time_spent_per_label_g2]
    ), axis=0)
    # additionally compute the average of percentage per group
    percentage_average_group1 = np.mean(np.array(
        [t[2] for t in list_of_time_spent_per_label_g1]
    ), axis=0)
    percentage_average_group2 = np.mean(np.array(
        [t[2] for t in list_of_time_spent_per_label_g2]
    ), axis=0)
    
    return (all_labels, 
            (frame_count_group1, percentage_average_group1), 
            (frame_count_group2, percentage_average_group2))
    


if __name__ == "__main__":
    # sanity check: compute_time_spent_per_label
    label = np.array([0]*25 + [1]*32 + [2]*34 + [0]*100)
    groups = [[0,1,2],
              [0,2], 
              [],
              [1],
              [2,1,0]]

    for g in groups:
        u_groups, framecounts, percentage = compute_time_spent_per_label(label=label, groups_to_check=g)
        print(f"g: {g}, u_groups: {u_groups}")
        print(f"framecounts: {framecounts}, percentage: {percentage}")