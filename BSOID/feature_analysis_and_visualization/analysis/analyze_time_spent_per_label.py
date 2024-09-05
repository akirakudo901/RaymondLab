# Author: Akira Kudo
# Created: 2024/04/02
# Last Updated: 2024/08/30

from typing import List

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
import tqdm

from bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
from feature_analysis_and_visualization.utils import find_runs, get_mousename

MOUSENAME = "mousename"
GROUPNAME = "groupname"
GROUPPREFIX = "group"
FRAMECOUNT, PERCENTAGE = "framecount", "percentage"

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
        framecount_in_row = np.zeros_like(all_unique_labels, dtype=np.int32)
        percentage_in_row = np.zeros_like(all_unique_labels, dtype=np.float32)
        framecount_in_row[indices] = raw_frame_counts
        percentage_in_row[indices] = percentage
        # form the rows
        framecount_row = [mousename] + framecount_in_row.tolist() + [grpname]
        percentage_row = [mousename] + percentage_in_row.tolist() + [grpname]
        # insert the rows into the data
        data_for_df.append(framecount_row); data_for_df.append(percentage_row)

    # create the dataframe
    df = pd.DataFrame(data=data_for_df,
                      columns=[MOUSENAME]+[f"{GROUPPREFIX}{lbl}" for lbl in all_unique_labels] + \
                              [GROUPNAME],
                      index=[FRAMECOUNT, PERCENTAGE] * len(data_per_mouse))
    if save_csv:
        df.to_csv(save_path)
    return df

def analyze_time_spent_per_label_between_groups(
        group1 : list,
        group2 : list,
        groupnames : List[str],
        significance : float,
        savedir : str,
        savename : str,
        data_is_normal : bool,
        equal_var : bool,
        label_groups : List[int]=None, 
        save_result : bool=True
        ):
    """
    Run analysis on aggregated data of groups, passed as lists to 
    the csvs of interest.
    
    Will examine:
    - The difference in average time spent between the same behavior 
      label among the group (Unpaired t / Mann-Whitney U)
    
    :param list group1: The first group as list of csv paths of labels, or 
    numpy arrays of labels.
    :param list group2: The second group as list of csv paths of labels, or 
    numpy arrays of labels.
    :param List[str] groupnames: A two-string list of the name of the groups 
    analyzed, used when saving the result.
    :param float significance: Level of significance used for comparsion.
    :param str savedir: Directory to which we save analysis results.
    :param str savename: Name of the file we save analysis results to.
    :param bool data_is_normal: Whether the data is normal or not. Use
    unpaired t-test if True, Mann-Whitney U test if False.
    :param bool equal_var: Whether the data has equal variance or not. 
    When data is normal, use unpaired t-test if True, and Welch's t-test
    if False. When data is not normal, has no effect.
    :param List[int] label_groups: A list of all group labels to 
    examine for comparion. Defaults to None - all labels.
    :param bool save_result: Whether to save the analysis results.
    
    :returns tuple (srtd_framecounts1, strd_percentages1): A tuple, the first
    dict mapping labels to the number of time the label was found in this array,
    and the second mapping labels to the percentage of the array this label covers.
    The first tuple is for group 1.
    :returns tuple (srtd_framecounts2, strd_percentages2): The same tuple as the first
    but for group 2.
    :returns dict stats: A dict mapping each label to the statistic found through 
    comparing the framecount of the two groups.
    :returns dict pvals: A dict mapping each label to the P-value found through 
    comparing the framecount of the two groups.
    """
    def compute_statsistics_for_group(label_array1 : list,
                                      label_array2 : list,
                                      groupnames : list,
                                      significance : float,
                                      savedir : str,
                                      savename : str,
                                      data_is_normal : bool,
                                      equal_var : bool, 
                                      save_result : bool):
        """
        Computes the stats for the two given data arrays.

        :param list label_array1: A list of label np.ndarrays for group 1.
        :param list label_array2: A list of label np.ndarrays for group 2.
        :param list groupnames: A two-string list of the name of the groups 
        analyzed, used when saving the result.
        :param float significance: Level of significance used for comparsion.
        :param str savedir: Directory to which we save analysis results.
        :param str savename: Name of the file we save analysis results to.
        :param bool data_is_normal: Whether the data is normal or not. Use
        unpaired t-test if True, Mann-Whitney U test if False.
        :param bool equal_var: Whether the data has equal variance or not. 
        When data is normal, use unpaired t-test if True, and Welch's t-test
        if False. When data is not normal, has no effect.
        :param bool save_result: Whether to save the analysis results.

        :returns tuple (srtd_framecounts1, strd_percentages1): A tuple, the first
        dict mapping labels to the number of time the label was found in this array,
        and the second mapping labels to the percentage of the array this label covers.
        The first tuple is for group 1.
        :returns tuple (srtd_framecounts2, strd_percentages2): The same tuple as the first
        but for group 2.
        :returns dict stats: A dict mapping each label to the statistic found through 
        comparing the framecount of the two groups.
        :returns dict pvals: A dict mapping each label to the P-value found through 
        comparing the framecount of the two groups.
        """
        all_results_txt, stats, pvals = "", {}, {}
        srtd_framecounts1, srtd_percentages1 = extract_framecounts_and_percentages(
            label_array1, label_groups)
        srtd_framecounts2, srtd_percentages2 = extract_framecounts_and_percentages(
            label_array2, label_groups)
        
        # compare the mean of the two framecount data for each key
        common_keys = np.unique(list(srtd_framecounts1.keys()) + 
                                list(srtd_framecounts2.keys()))
        for k in common_keys:
            data1, data2 = srtd_framecounts1[k], srtd_framecounts2[k]
            
            statistic, pvalue, result_txt = compare_independent_sample_means(
                data1=data1, data2=data2, groupnames=groupnames, 
                significance=significance, savedir=None, savename=None, 
                data_is_normal=data_is_normal, 
                equal_var=equal_var, save_result=False
                )
            all_results_txt += f"Label: {k}."
            all_results_txt += result_txt
            stats[k] = statistic; pvals[k] = pvalue
        
        if save_result:
            savepath = os.path.join(savedir, savename)
            with open(savepath, 'w') as f:
                f.write(all_results_txt)

        return (srtd_framecounts1, srtd_percentages1), (srtd_framecounts2, srtd_percentages2), \
                stats, pvals
    
    if type(group1[0]) == str: 
        group1_labels = [read_BSOID_labeled_csv(csv)[0] for csv in group1]
    else: group1_labels = group1

    if type(group2[0]) == str: 
        group2_labels = [read_BSOID_labeled_csv(csv)[0] for csv in group2]
    else: group2_labels = group2
    
    # compute statistics
    (framecount1, percentage1), (framecount2, percentage2), stats, pvals = \
        compute_statsistics_for_group(label_array1=group1_labels, label_array2=group2_labels, 
                                  groupnames=groupnames, significance=significance, 
                                  savedir=savedir, savename=savename, 
                                  data_is_normal=data_is_normal, equal_var=equal_var,
                                  save_result=save_result)
    
    return (framecount1, percentage1), (framecount2, percentage2), stats, pvals

def analyze_average_behavior_snippet_length_per_label_per_group(
        group1 : list,
        group2 : list,
        groupnames : List[str],
        significance : float,
        savedir : str,
        savename : str,
        data_is_normal : bool,
        equal_var : bool,
        label_groups : List[int]=None, 
        save_result : bool=True
        ):
    """
    Takes in two lists of labels with corresponding names, extracting how the 
    length of bouts of each label is varied for each mouse.
    The mean bout length & standard deviation of bout lengths is computed for each 
    mouse. These values are then compared using either the Unpaired T-test (if 
    data_is_normal is True) or the Mann-Whitney U test (if data_is_normal is False).

    :param list group1: The first group as list of csv paths of labels, or 
    numpy arrays of labels.
    :param list group2: The second group as list of csv paths of labels, or 
    numpy arrays of labels.
    :param List[str] groupnames: A two-string list of the name of the groups 
    analyzed, used when saving the result.
    :param float significance: Level of significance used for comparsion.
    :param str savedir: Directory to which we save analysis results.
    :param str savename: Name of the file we save analysis results to.
    :param bool data_is_normal: Whether the data is normal or not. Use
    unpaired t-test if True, Mann-Whitney U test if False.
    :param bool equal_var: Whether the data has equal variance or not. 
    When data is normal, use unpaired t-test if True, and Welch's t-test
    if False. When data is not normal, has no effect.
    :param List[int] label_groups: A list of all group labels to 
    examine for comparion. Defaults to None - all labels.
    :param bool save_result: Whether to save the analysis results.
    """
    if len(groupnames) != 2:
        raise Exception(f"groupnames must be of length 2, but was instead {len(groupnames)} long...")
    
    if type(group1[0]) == str: 
        group1_labels = [read_BSOID_labeled_csv(csv)[0] for csv in group1]
    else: group1_labels = group1

    if type(group2[0]) == str: 
        group2_labels = [read_BSOID_labeled_csv(csv)[0] for csv in group2]
    else: group2_labels = group2

    def process_labels_by_groups(group_label : list):
        group_means, group_sd = {}, {}
        for label in group_label:
            mean_dict, sd_dict = compute_bout_length_mean_and_sd_per_mouse(label)
            for unique_label in mean_dict.keys():
                ul_mean_dict = group_means.get(unique_label, [])
                ul_mean_dict.append(mean_dict[unique_label])
                group_means[unique_label] = ul_mean_dict
            for unique_label in sd_dict.keys():
                ul_sd_dict = group_sd.get(unique_label, [])
                ul_sd_dict.append(sd_dict[unique_label])
                group_sd[unique_label] = ul_sd_dict
        return group_means, group_sd
    
    group1_means, group1_sd = process_labels_by_groups(group1_labels)
    group2_means, group2_sd = process_labels_by_groups(group2_labels)

    unique_labels = set(group1_means.keys()).union(
        set(group1_sd.keys()), 
        set(group2_means.keys()),
        set(group2_sd.keys())
        )
    
    all_mean_results_txt, all_sd_results_txt = "", ""
    mean_stats, mean_pvals, sd_stats, sd_pvals = {}, {}, {}, {}

    for lbl in unique_labels:
        if label_groups is not None and lbl not in label_groups: continue
        # compare means first
        statistic, pvalue, mean_result_txt = compare_independent_sample_means(
            data1=group1_means[lbl], 
            data2=group2_means[lbl], 
            groupnames=groupnames,
            significance=significance,
            savedir=savedir,
            savename=savename,
            data_is_normal=data_is_normal,
            equal_var=equal_var,
            save_result=False
            )
        
        all_mean_results_txt += f"Label: {lbl}."
        all_mean_results_txt += mean_result_txt
        mean_stats[lbl] = statistic; mean_pvals[lbl] = pvalue

        # then compare standard deviations
        statistic, pvalue, sd_result_txt = compare_independent_sample_means(
            data1=group1_sd[lbl], 
            data2=group2_sd[lbl], 
            groupnames=groupnames,
            significance=significance,
            savedir=savedir,
            savename=savename,
            data_is_normal=data_is_normal,
            equal_var=equal_var,
            save_result=False
            )
        all_sd_results_txt += f"Label: {lbl}."
        all_sd_results_txt += sd_result_txt
        sd_stats[lbl] = statistic; sd_pvals[lbl] = pvalue
    
    if save_result:
        mean_savepath = os.path.join(savedir, savename.replace('.', 'Mean.'))
        with open(mean_savepath, 'w') as f:
            f.write(all_mean_results_txt)
        
        sd_savepath = os.path.join(savedir, savename.replace('.', 'SD.'))
        with open(sd_savepath, 'w') as f:
            f.write(all_sd_results_txt)

def extract_framecounts_and_percentages(label_array : list,
                                        label_groups : list=None):
    """
    Given a list of label arrays, extract the frame counts and 
    percentage over all frames for each label specified in label_groups.

    :param list label_array: A list of labels we process.
    :param list label_groups: A list of all group labels to 
    examine. Defaults to None - all labels.
    
    :returns dict srtd_framecounts: A dict mapping labels to the number of 
    time the label was found in this array. The keys are ascendingly sorted.
    :returns dict srtd_percentages: A dict mapping labels to the percentage
    of the array this label covers. The keys are ascendingly sorted.
    """
    framecounts, percentages = {},{}
    for lbl in label_array:
        unique_labels, framecount, percentage = compute_time_spent_per_label(
            label=lbl, label_groups=label_groups)
        for idx, unique_l in enumerate(unique_labels):
            fc, perc = framecount[idx].item(), percentage[idx].item()
            framecounts[unique_l] = framecounts.get(unique_l, []) + [fc]
            percentages[unique_l] = percentages.get(unique_l, []) + [perc]
    # resort the label orders
    sorted_keys = list(framecounts.keys()); sorted_keys.sort()
    srtd_framecounts, srtd_percentages = {},{}
    for k in sorted_keys:
        srtd_framecounts[k] = framecounts[k]
        srtd_percentages[k] = percentages[k]
    
    return srtd_framecounts, srtd_percentages

def compare_independent_sample_means(data1 : np.ndarray, 
                                     data2 : np.ndarray, 
                                     groupnames : list,
                                     significance : float,
                                     savedir : str,
                                     savename : str,
                                     data_is_normal : bool,
                                     equal_var : bool,
                                     save_result : bool=True):
    """
    Compare two independent sample means, using the Unpaired T-test if the data
    is normal, or the Mann-Whitney U test if not.
    :param np.ndarray data1: Data array 1.
    :param np.ndarray data2: Data array 2.
    :param list groupnames: The name of the two data groups examined, in order 1 and 2.
    :param float significance: Significance level for comparison.
    :param str savedir: Directory to which the result is saved.
    :param str savename: Name of the saved file.
    :param bool data_is_normal: Whether the data can be assumed normal. Unpaired
    T-test if True, Mann-Whitney U test if False.
    :param bool equal_var: Whether the data has equal variances. If data is normal, 
    Unpaired t-test when equal_var is True, Welch's t-test otherwise. If data is 
    not normal, no effect.
    :param bool save_result: Whether to save the result. Defaults to True.

    :returns float statistic: The statistic value from the examination.
    :returns float pvalue: The P value obtained from the comparison.
    :returns str all_results_txt: A text containing analysis result.
    """
    all_results_txt = ""

    if data_is_normal:
        test_res = ttest_ind(data1, data2, equal_var=equal_var)
    else:
        test_res = mannwhitneyu(data1, data2, use_continuity=True)
    
    significant_result = test_res.pvalue < significance
    result_txt = f"""
Examining groups: {groupnames[0]} (n={len(data1)}); {groupnames[1]} (n={len(data2)}).
Means for {groupnames[0]} and {groupnames[1]} are: {np.mean(data1)}; {np.mean(data2)}.
Result is {'significant!' if significant_result else 'not significant'}: \
{test_res.pvalue} {'<' if significant_result else '>'} {significance}.
"""
    print(result_txt)
    all_results_txt += result_txt
    
    if save_result:
        save_to = os.path.join(savedir, savename)
        print(f"Saving result for groups {', '.join(groupnames)} to: {save_to}...", end="")
        with open(save_to, 'w') as f:
            f.write(all_results_txt)
        print("SUCCESSFUL!")

    return test_res.statistic, test_res.pvalue, all_results_txt

def compute_bout_length_mean_and_sd_per_mouse(label : np.ndarray):
    """
    Identifies the lengths of every bout of behavior, and computes both
    the mean length of such bouts and their standard deviation, for 
    each unique label in the given label array.

    :param np.ndarray label: The provided label array.
    :return dict run_length_means: A dictionary mapping each unique label in 
    the label array to the average length of bouts of that label.
    :return dict run_length_sd: A dictionary mapping each unique label in 
    the label array to the standard deviation of length of bouts of that label.
    """
    run_length_means, run_length_sd = {}, {}

    run_values, _, run_lengths = find_runs(label)

    for unique_label in np.unique(label):
        ul_run_lenghts = run_lengths[unique_label == run_values]
        run_length_means[unique_label.item()] = np.mean(ul_run_lenghts)
        run_length_sd[unique_label.item()] = np.std(ul_run_lenghts)
    
    return run_length_means, run_length_sd

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