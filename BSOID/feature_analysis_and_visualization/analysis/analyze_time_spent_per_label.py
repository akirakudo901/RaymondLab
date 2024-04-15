# Author: Akira Kudo
# Created: 2024/04/02
# Last Updated: 2024/04/15

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
    unique_groups = np.sort(np.unique(label)).astype(np.int64)
    # depending on groups_to_check, filter which group label we consider
    if groups_to_check is not None and len(groups_to_check) != 0:
        keep_these_indices = np.isin(unique_groups, np.array(groups_to_check))
        unique_groups = unique_groups[keep_these_indices]
    # compute the values of interest
    raw_frame_counts = []
    for unique_lbl in unique_groups:
        count = np.sum(label == unique_lbl)
        raw_frame_counts.append(count)

    raw_frame_counts = np.array(raw_frame_counts).astype(np.int64)
    percentage = raw_frame_counts / np.sum(raw_frame_counts) * 100
    
    return unique_groups, raw_frame_counts, percentage


def analyze_time_spent_per_label_between_groups(
        group1 : List[str],
        group2 : List[str],
        groups_to_check : List[int]=None,
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
                label=lbl, groups_to_check=groups_to_check)
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