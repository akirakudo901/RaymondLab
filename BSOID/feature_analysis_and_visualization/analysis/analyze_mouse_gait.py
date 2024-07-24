# Author: Akira Kudo
# Created: 2024/03/31
# Last updated: 2024/07/16

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.linear_model import LinearRegression
import yaml

from ..visualization.visualize_mouse_gait import filter_nonpawrest_motion, read_stepsize_yaml, select_N_locomotion_sequences, COL_BODYPART, COL_START, COL_X_AVG, COL_Y_AVG, STEPSIZE_MIN

"""
Take a single csv. From it, we can:
1) Extract all locomotion bouts
2) Visualize info:
 - How many locomotion bouts there are
 - How many of each length of locomotion bouts there are
 - What's the distribution of the distance between two locomotion bouts (curious if bouts are broken down but close together)
 - What's the distribution of locomotion for every 5 min interval (just curious)
3) Extract the paw stopping based on speed
4) For every paw stopping identified as single bout, find the average paw position within that stop
5) Graph such paw positions over time in:
 - a static plot
 - video, to confirm of its correctness
6) Save this data into a csv, and:
 - calculate the distance between each paw movement
 - calculate the distance between each body part
 and so on.
7) Visualize:
- distance traveled in a single bout
- max / min / average speed of those single bouts
- frequency of bouts with different speeds
8) Maybe do analysis based on those variables
"""

LOCOMOTION_LABELS = [38]
MOVEMENT_THRESHOLD = 0.5

def analyze_mouse_gait(df : pd.DataFrame, 
                       label : np.ndarray, 
                       bodyparts : list, 
                       savedir : str,
                       savename : str,
                       locomotion_label=LOCOMOTION_LABELS, 
                       length_limits : tuple=(None,None),
                       save_result : bool=True
                       ):
    
    # 1) Extract all locomotion bouts
    starts, ends, _ = select_N_locomotion_sequences(
        label=label, N=float('inf'), locomotion_labels=locomotion_label, 
        length_limits=length_limits
        )
    
    # 2) Visualize info:
    # - How many locomotion bouts there are
    # - How many of each length of locomotion bouts there are
    # - What's the distribution of the distance between two locomotion bouts (curious if bouts are broken down but close together)
    # - What's the distribution of locomotion for every 5 min interval (just curious)

    # << THIS PART, MAYBE CHECK visualize_mouse_gait.py

    # quantify the distance between consecutive steps
    # when doing so, align the direction with the average of all paw positions, 
    # weighted in the number of right & left paws.
    
    # first, get the distance between consecutive steps
    df, avg_df = filter_nonpawrest_motion(
        df=df, label=label, show_nonpaw=True, threshold=MOVEMENT_THRESHOLD, 
        locomotion_labels=locomotion_label, average_pawrest=True
        )
    
    unique_bpts = np.unique(avg_df[COL_BODYPART])

    data = {}

    for start, end in zip(starts, ends):
        start, end = start.item(), end.item()
        for bpt in unique_bpts:
            if bpt not in bodyparts: continue
            
            steps = avg_df[(avg_df[COL_START] >= start) & 
                           (avg_df[COL_START] <=   end) & 
                           (avg_df[COL_BODYPART] == bpt)]
            x_diff = np.abs(np.diff(steps[COL_X_AVG]))
            y_diff = np.abs(np.diff(steps[COL_Y_AVG]))
            diff = np.sqrt(x_diff**2 + y_diff**2)

            # create a new entry to data
            new_entry = data.get(start, {})
            
            bpt_entry = new_entry.get(bpt, {})
            bpt_entry["x_diff"] = x_diff.tolist()
            bpt_entry["y_diff"] = y_diff.tolist()
            bpt_entry["diff"] = diff.tolist()

            new_entry[bpt] = bpt_entry
            new_entry["end"] = end
            data[start] = new_entry

    if save_result:
        yaml_result = yaml.dump(data)
        with open(os.path.join(savedir, savename), 'w') as f:
            f.write(yaml_result)

    return data

def extract_average_paw_rests(df : pd.DataFrame, 
                              label : np.ndarray, 
                              bodyparts : list, 
                              savedir : str,
                              savename : str,
                              locomotion_label=LOCOMOTION_LABELS, 
                              length_limits : tuple=(None,None),
                              save_result : bool=True):
    # extract all locomotion bouts
    starts, ends, _ = select_N_locomotion_sequences(
        label=label, N=float('inf'), locomotion_labels=locomotion_label, 
        length_limits=length_limits
        )
    
    
    # quantify the distance between consecutive steps
    # when doing so, align the direction with the average of all paw positions, 
    # weighted in the number of right & left paws.
    
    # first, get the distance between consecutive steps
    df, avg_df = filter_nonpawrest_motion(
        df=df, label=label, show_nonpaw=True, threshold=MOVEMENT_THRESHOLD, 
        locomotion_labels=locomotion_label, average_pawrest=True
        )
    
    unique_bpts = np.unique(avg_df[COL_BODYPART])

    data = {}

    for start, end in zip(starts, ends):
        start, end = start.item(), end.item()
        for bpt in unique_bpts:
            if bpt not in bodyparts: continue
            
            steps = avg_df[(avg_df[COL_START] >= start) & 
                           (avg_df[COL_START] <=   end) & 
                           (avg_df[COL_BODYPART] == bpt)]
            x_diff = np.abs(np.diff(steps[COL_X_AVG]))
            y_diff = np.abs(np.diff(steps[COL_Y_AVG]))
            diff = np.sqrt(x_diff**2 + y_diff**2)

            # create a new entry to data
            new_entry = data.get(start, {})
            
            bpt_entry = new_entry.get(bpt, {})
            bpt_entry["x_diff"] = x_diff.tolist()
            bpt_entry["y_diff"] = y_diff.tolist()
            bpt_entry["diff"] = diff.tolist()

            new_entry[bpt] = bpt_entry
            new_entry["end"] = end
            data[start] = new_entry

    if save_result:
        yaml_result = yaml.dump(data)
        with open(os.path.join(savedir, savename), 'w') as f:
            f.write(yaml_result)

    return data

def extract_average_line_between_paw(pawrest_df : pd.DataFrame,
                                    #  bodyparts : list, 
                                     savedir : str,
                                     savename : str):
    # TODO AMELIORATE!
    result = "bodypart,slope,intercept\n"
    
    for bpt in ['rightforepaw', 'leftforepaw', 'righthindpaw', 'lefthindpaw']:
        pawrest_x = pawrest_df[pawrest_df[COL_BODYPART] == bpt][COL_X_AVG].to_numpy().reshape(-1,1)
        pawrest_y = pawrest_df[pawrest_df[COL_BODYPART] == bpt][COL_Y_AVG].to_numpy().reshape(-1,1)
        reg = LinearRegression()
        reg.fit(X=pawrest_x, y=pawrest_y)
        slope, intercept = reg.coef_[0][0], reg.intercept_[0]
        print(f"For bpt {bpt}, slope: {slope}; intercept: {intercept}")
        result += f"{bpt},{slope},{intercept}\n"
    
    with open(os.path.join(savedir, savename), 'w') as f:
        f.write(result)

def analyze_mouse_stepsize_per_mousegroup_from_yamls(
        yamls1 : list,
        yamls2 : list,
        groupnames : list,
        bodyparts : list,
        # uses unpaired-t if true, mann whitney u if false
        data_is_normal : bool=True,
        significance : float=0.05,
        cutoff : float=STEPSIZE_MIN,
        save_result : bool=True,
        save_to : str=None
        ):
    """
    Takes in two groups of yaml files holding information of step sizes,
    computing whether the mean stepsizes of different body parts are different 
    or not using either unpaired t-test with normally distributed data or 
    Mann-Whitney U test with non-normally distributed data. 
    Uses the given significance threshold.

    :param list yamls1: List of yamls holding stepsize data for the first group.
    :param list yamls2: List of yamls holding stepsize data for the second group.
    :param list groupnames: The name of the two groups. Its length must be two.
    :param list bodyparts: The body parts we compare.
    :param bool data_is_normal: Whether to use unpaired t-test if true, or 
    Mann-Whitney U if false, defaults to True
    :param float significance: Significance threshold, defaults to 0.05
    :param float cutoff: A cutoff for the minimum step size considered a step, 
    defaults to STEPSIZE_MIN
    :param bool save_result: Whether to save analysis result in a text file, 
    defaults to True
    :param str save_to: Full path including file name to save the text file, 
    defaults to None

    :return float statsitic: The test statistic.
    :return float pvalue: Value of significance.
    """
    dicts1 = [read_stepsize_yaml(yaml) for yaml in yamls1]
    dicts2 = [read_stepsize_yaml(yaml) for yaml in yamls2]

    statistic, pvalue = analyze_mouse_stepsize_per_mousegroup_from_dicts(
        dicts1=dicts1, dicts2=dicts2, groupnames=groupnames, 
        bodyparts=bodyparts, data_is_normal=data_is_normal, 
        significance=significance, cutoff=cutoff, save_result=save_result,
        save_to=save_to)
    
    return statistic, pvalue

def analyze_mouse_stepsize_per_mousegroup_from_dicts(
        dicts1 : list,
        dicts2 : list,
        groupnames : list,
        bodyparts : list,
        # uses unpaired-t if true, mann whitney u if false
        data_is_normal : bool=True,
        significance : float=0.05,
        cutoff : float=STEPSIZE_MIN,
        save_result : bool=True,
        save_to : str=None
        ):
    """
    Takes in two groups of dictionaries holding information of step sizes,
    computing whether the mean stepsizes of different body parts are different 
    or not using either unpaired t-test with normally distributed data or 
    Mann-Whitney U test with non-normally distributed data. 
    Uses the given significance threshold.

    :param list dicts1: List of dictionaries holding stepsize data for the first group.
    :param list dicts2: List of dictionaries holding stepsize data for the second group.
    :param list groupnames: The name of the two groups. Its length must be two.
    :param list bodyparts: The body parts we compare.
    :param bool data_is_normal: Whether to use unpaired t-test if true, or 
    Mann-Whitney U if false, defaults to True
    :param float significance: Significance threshold, defaults to 0.05
    :param float cutoff: A cutoff for the minimum step size considered a step, 
    defaults to STEPSIZE_MIN
    :param bool save_result: Whether to save analysis result in a text file, 
    defaults to True
    :param str save_to: Full path including file name to save the text file, 
    defaults to None

    :return float statsitic: The test statistic.
    :return float pvalue: Value of significance.
    """

    if len(groupnames) != 2:
        raise Exception(f"groupnames has to be of length 2 but is {len(groupnames)}...")
    
    all_results_txt = ""
    
    all_stepsizes1 = aggregate_stepsize_per_body_part(
        dictionaries=dicts1, bodyparts=bodyparts, cutoff=cutoff)
    all_stepsizes2 = aggregate_stepsize_per_body_part(
        dictionaries=dicts2, bodyparts=bodyparts, cutoff=cutoff)
        
    # for every body part
    for bpt in bodyparts:
        filtered_stepsizes1 = remove_outlier_data(
            np.array(all_stepsizes1[bpt]))
        filtered_stepsizes2 = remove_outlier_data(
            np.array(all_stepsizes2[bpt]))

        if data_is_normal:
            test_res = ttest_ind(filtered_stepsizes1, filtered_stepsizes2)
        else:
            test_res = mannwhitneyu(filtered_stepsizes1, filtered_stepsizes2, 
                                    use_continuity=True)
    
        significant_result = test_res.pvalue < significance
        result_txt = f"""
Examining groups: {groupnames[0]} (n={len(filtered_stepsizes1)}); {groupnames[1]} (n={len(filtered_stepsizes2)}) for {bpt}.
Result is {'significant!' if significant_result else 'not significant'}: \
{test_res.pvalue} {'<' if significant_result else '>'} {significance}.
"""
        print(result_txt)
        all_results_txt += result_txt
    
    if save_result:
        print(f"Saving result for groups {', '.join(groupnames)} to: {save_to}...", end="")
        with open(save_to, 'w') as f:
            f.write(all_results_txt)
        print("SUCCESSFUL!")

    return test_res.statistic, test_res.pvalue

def extract_time_difference_between_consecutive_left_right_contact(
        df : pd.DataFrame, 
        label : np.ndarray,
        sequences : list,
        savedir : str,
        savename : str,
        comparison_pairs : list,
        ignore_close_paw : float=None,
        locomotion_label : list=LOCOMOTION_LABELS,
        save_result : bool=True,
        ):
    """
    Extracts the difference in time betwen the consecutive landing of specified body parts. 
    Consecutive landings are identified as steps taken temporarily in succession, where one body part 
    lands between two landings of the other body part.
    Any time one body parts lands twice during which the other does not land at all, the slot is replaced
    with nan.
    E.g. If bpt1 lands at [1,5,10,12,15] and bpt2 lands at [2,7,11,16], consecutive landings are:
    FOR BPT1 TO BPT2: [2-1,  7-5, 11-10, ?-12, 16-15] = [1,2,1,nan,1]
    FOR BPT2 TO BPT1: [5-2, 10-7, 12-11, ?-16]        = [3,3,1,nan]

    :param pd.DataFrame df: Data frame holding DLC data.
    :param np.ndarray label: B-SOID labels to identify locomotion.
    :param list sequences: A list of tuples of two floats, each tuple being
    the start and end of a locomotion sequence we want to analyze.
    :param str savedir: Directory to where we save results.
    :param str savename: Name of the saved file.
    :param list comparison_pairs: Pairs of body parts to compare. Must be valid body 
    parts belonging to df.columns.
    :param float ignore_close_paw: How close consecutive paws must be in order
    to be ignored as a single paw, defaults to None
    :param list locomotion_label: A list of labels indicating locomotion for B-SOID
    labels, defaults to LOCOMOTION_LABELS
    :param bool save_result: Whether to save the result, defaults to True
    """
       
    # get the distance between consecutive steps & get the averaged data frame
    df, avg_df = filter_nonpawrest_motion(
        df=df, label=label, show_nonpaw=True, threshold=MOVEMENT_THRESHOLD, 
        locomotion_labels=locomotion_label, average_pawrest=True,
        ignore_close_paw=ignore_close_paw
        )
    
    result = {}

    # for one "start" and the next "start" for a paw, find any "start" of the other paw
    # that exists in between
    # then take the difference between the first paw's start and the second's
    def find_opposite_steps(main_stepstarts : np.ndarray, 
                            opposite_stepstarts : np.ndarray):
        """
        Given two sequences of values, iterate through the first
        sequence: 
        1 - choosing one entry and the one immediately after it
        2 - finding a value from the second sequence that resides between the
            two entries
        We make a list out of this process. 
        If more than one value is identified in step 2, the smallest value is 
        added to the returned list. If no value is identified, np.nan is added.

        :param np.ndarray main_stepstarts: The main sequence we iterate through.
        :param np.ndarray opposite_stepstarts: The sequence holding step starts
        from which we find the 'opposite' steps.
        :return list: A list of 'opposite steps' that match 1-to-1 with the first
        sequence passed.
        """
        opposite_steps = []
        for step_idx, step_start in enumerate(main_stepstarts):
            if step_idx == (len(main_stepstarts) - 1):
                next_start = np.inf
            else:
                next_start = main_stepstarts[step_idx+1]
            
            opposite_step_start = opposite_stepstarts[
                (opposite_stepstarts > step_start) & (opposite_stepstarts < next_start)
                ]
            if len(opposite_step_start) == 0:
                opposite_step_start = np.nan
            else:
                opposite_step_start = np.min(opposite_step_start)
            opposite_steps.append(opposite_step_start)
        return opposite_steps
    
    all_results = {}

    # for each identified locomotion bouts
    for (start, end) in sequences:
        result = {}
        for pair in comparison_pairs:
            # identify all 'steps' belong to this locomotion sequence
            steps1 = avg_df[(avg_df[COL_START] >= start) &
                            (avg_df[COL_START] <=   end) &
                            (avg_df[COL_BODYPART] == pair[0])]
            steps2 = avg_df[(avg_df[COL_START] >= start) &
                            (avg_df[COL_START] <=   end) &
                            (avg_df[COL_BODYPART] == pair[1])]
            step_starts1 = steps1[COL_START].to_numpy()
            step_starts2 = steps2[COL_START].to_numpy()
            
            # for each entry in step_starts1, we find the next start value
            # as well -> then we find the minimum value in step_starts2 that 
            # resides between this start and the next, keeping record
            opposite_steps_forward = find_opposite_steps(
                main_stepstarts=step_starts1, opposite_stepstarts=step_starts2
                )
            opposite_steps_backward = find_opposite_steps(
                main_stepstarts=step_starts2, opposite_stepstarts=step_starts1
                )
            
            time_diff_forward  = np.array(opposite_steps_forward) - step_starts1
            time_diff_backward = np.array(opposite_steps_backward) - step_starts2
            
            result[f'{pair[0]}_{pair[1]}'] = time_diff_forward.tolist()
            result[f'{pair[1]}_{pair[0]}'] = time_diff_backward.tolist()
        result["end"] = end
        all_results[start] = result
    
    if save_result:
        yaml_result = yaml.safe_dump(all_results)
        with open(os.path.join(savedir, savename), 'w') as f:
            f.write(yaml_result)       


# HELPERS

def remove_outlier_data(data : np.array):
    """
    Removes any clear outliers that are beyond either
    1.5 interquartile range above 75 percentile / below 25 percentile.

    :param np.ndarray data: Data from which we exclude outliers.

    :returns np.ndarray filtered: A filtered numpy array.
    """
    IQR_MULTIPLIER = 1.5
    # exclude clear outliers beyond 1.5 IQR above 75 & below 25 percentile
    percentiles = np.percentile(data, [25, 75])
    q_one, q_three = percentiles[0].item(), percentiles[1].item()
    iqr = q_three - q_one
    
    filtered = data[(data < (q_three + iqr*IQR_MULTIPLIER)) & 
                    (data > (q_one   - iqr*IQR_MULTIPLIER))]
    return filtered

def identify_curved_trajectory(X : np.ndarray,
                               Y : np.ndarray, 
                               windowsize : int=20, 
                               threshold : float=np.pi/4):
    """
    Given the trajectory of a body part (presumably that of the tail base),
    identifies whether there is a gross change in direction of the trajectory, 
    as well as if there is one, where it happens.

    :param np.ndarray X: The x coordinates of the trajectory to be analyzed.
    :param np.ndarray Y: The y coordinates of the trajectory to be analyzed.
    :param int windowsize: The size of a 'window' used to calculate the average
    direction of a segment, to be compared with others.
    :param float threshold: Threshold of how much the trajectory must change to be
    recognized as change in direction. In radians, defaulting to pi/4.

    :returns np.ndarray change_at: Indices at which the change in absolute direction
    between consecutive windows exceeded threshold. N being in 'change_at' signifies 
    that the absolute change in direction between windows starting at N and N+1 exceeded
    threshold.
    """
    # we construct 'windows' of size windowsize at stepsize intervals
    # and compare its first and last positions
    windowstart_x, windowend_x = X[:-windowsize], X[windowsize:]
    windowstart_y, windowend_y = Y[:-windowsize], Y[windowsize:]
    delta_x = windowend_x - windowstart_x
    delta_y = windowend_y - windowstart_y
    # compute the direction for each such window
    direction = np.arctan2(delta_y, delta_x)
    # if the absolute change in direction is greater than threshold, mark that
    direction_start, direction_end = direction[:-windowsize], direction[windowsize:]
    change_at = np.nonzero(np.abs(direction_end - direction_start) > threshold)[0]
    if len(change_at) != 0:
        change_at += windowsize - 1
        # we take any consecutive such points and condense them to their median value
        consecutive = np.split(change_at, np.where(np.diff(change_at) != 1)[0]+1)
        change_at = np.concatenate([np.floor(np.mean(group, keepdims=True)) 
                                    for group in consecutive]).astype(np.int)
    return change_at

def filter_stepsize_dict_by_locomotion_to_use(dict : dict, 
                                              mousename : str,
                                              locomotion_to_use : dict):
    """
    Given a dictionary of step sizes for a single mouse and its mouse name,
    fetch info on which corresponding step sequences to use from another 
    dictionary ('locomotion_to_use') and return a dict only holding those
    locomotions to use.
    'locomotion_to_use' will be a dict of mousename : [list of list of ints]
    where each list of integer specifies the statr and end of a sequence, like 
    the following: 
    301532m3 : [[0, 10], [25, 73]]
    specifies that for mouse 301532m3, we use sequences 0~10 and 25~73.

    :param dict dict: Dictionary holding step size data, likely taken from a yaml.
    :param str mousename: Name of the mouse, to be found as key in locomotion_to_use.
    :param dict locomotion_to_use: A dictionary mapping mice names to their list of
    locomotion sequences to use. Those matching in start and end frames are used.

    :returns dict used_locomotions: Dictionary holding step size data specifically for 
    those locomotion sequences specified to be used.
    """
    try:
        loc_start_end = locomotion_to_use[mousename]
    except KeyError:
        raise Exception(f"Mouse {mousename} not found in locomotion_to_use ...")
    
    used_locomotions = {}
    if loc_start_end is None: return {}

    for start, end in loc_start_end:
        # check an entry with key as 'start' exists
        if dict.get(start, None) is not None:
            # if it does, check if the 'end' entry matches
            if dict[start]["end"] == end:
                # if they do, add them to the locomotions to be returned
                used_locomotions[start] = dict[start]
    
    return used_locomotions

def aggregate_stepsize_per_body_part(dictionaries : list, 
                                     bodyparts : list, 
                                     cutoff : float=0):
    """
    Given a list of dictionaries of step sizes, returns all step sizes for a 
    specific body part that are above cutoff as a single list - returning a single 
    dictionary of "body part" : "list of step sizes".

    :param list dicts: A list of dictionaries holding step sizes.
    :param list bodyparts: List of body parts for which we aggregate step sizes.
    :param float cutoff: Minimum step size to be included, defaults to 0
    
    :return dict: A dictionary mapping each body part to all of the corresponding 
    step sizes above cutoff, each put together into a list.
    """
    aggregated = {}
    for bpt in bodyparts:
        all_stepsizes = []
        for d in dictionaries:
            for sequence in d.values():
                filt_stepsize = [stepsize for stepsize in sequence[bpt]['diff']
                                if stepsize > cutoff]
                all_stepsizes.extend(filt_stepsize)
        aggregated[bpt] = all_stepsizes
    return aggregated