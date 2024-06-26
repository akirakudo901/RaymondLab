# Author: Akira Kudo
# Created: 2024/03/31
# Last updated: 2024/06/26

import os

from math import sqrt
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..utils import find_runs, get_mousename, process_upper_and_lower_limit

LOCOMOTION_LABELS = [38]
MOVEMENT_THRESHOLD = 0.75
STEPSIZE_MIN = 7 # a cutoff for a stepsize that is considered the same position

# column names for an averaged paw movement data frame
COL_BODYPART = 'bodypart'
COL_START = 'start'
COL_LENGTH = 'length'
COL_X_AVG = 'x_avg'
COL_Y_AVG = 'y_avg'

def visualize_mouse_gait_speed(df : pd.DataFrame,
                                label : np.ndarray, 
                                bodyparts : list, 
                                length_limits : tuple=(None, None),
                                locomotion_label : list=LOCOMOTION_LABELS, 
                                plot_N_runs : int=float("inf")):
    """
    Visualizes speed of the mouse gait by taking in data & label,
    identifying sequences of gait in the data, and then
    aligning all data to start from the beginning of first 
    paw movement.

    :param pd.DataFrame df: Dataframe holding x, y data of bodyparts.
    :param np.ndarray label: Numpy array holding labels of the same dataframe.
    :param list bodyparts: A list of body part names to visualize for.
    :param tuple length_limits: In form (low, high), indicates what range 
    of bout lengths should be used for visualization. A None at either 
    position indicates no limit on the upper / lower bound. 
    Defaults to (None, None), no restriction.
    :param int locomotion_label: Specifies which labels corresponds to 
    the locomotion group. Defaults to LOCOMOTION_LABELS.
    :param int plot_N_runs: The number of runs we plot. Defaults to all.
    """
    locomotion_idx, locomotion_lengths = find_locomotion_sequences(
        label=label, locomotion_label=locomotion_label, length_limits=length_limits)
    
    _, axes = plt.subplots(len(bodyparts), 1)
    align_with = {}

    for bpt_idx, bpt in enumerate(bodyparts):
        x, y = df[bpt, 'x'].to_numpy(), df[bpt, 'y'].to_numpy()
        movement = np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y)))
        
        # render the gate into a timeseries
        ax = axes[bpt_idx]
        
        for i, run_start in enumerate(locomotion_idx):
            if i >= plot_N_runs: break
            run_end = run_start + locomotion_lengths[i] - 1
            run = movement[run_start:run_end+1]
            run_duration= f"{run_start}~{run_end}"
            # obtain where the paw movement starts and ends based on threshold
            # we align every bodypart to the first bpt's paw movement
            if bpt_idx == 0:
                move_start, move_end, move_middle = find_paw_movements(
                    run, threshold=MOVEMENT_THRESHOLD)
                if len(move_middle) != 0:
                    align_with[run_duration] = move_middle[0]
                else:
                    align_with[run_duration] = 0
            # we plot every run while shifting them so that 0 is the first paw movement
            ax.plot(range(-align_with[run_duration], len(run) - align_with[run_duration]), 
                    run, label=run_duration)

        ax.set_title(f"{bpt}")
        if bpt_idx == len(bodyparts) - 1:
            ax.set_xlabel('Time from first paw movement')
        ax.set_ylabel(f'Distance')
        ax.legend()
    plt.suptitle("Locomotion Sequences Distance")
    plt.show()


def visualize_mouse_paw_rests_in_locomomotion(
        df : pd.DataFrame,
        label : np.ndarray, 
        bodyparts : list,
        savedir : str,
        savename : str,
        length_limits : tuple=(None, None),
        locomotion_label : list=LOCOMOTION_LABELS, 
        plot_N_runs : int=float("inf"),
        threshold : float=MOVEMENT_THRESHOLD,
        save_figure : bool=True,
        show_figure : bool=True
        ):
    """
    Identifies 'plot_N_runs' sequences where label happens consecutively within 
    range of length_limits, finding the position of paws at rest during these sequences
    by observing any lack of movement (speed between frames being below threshold).
    Renders such paw rests as well as the trajectory of non-paw body parts as a 
    single figure per such sequence. Which body part are rendered can be specified.

    :param pd.DataFrame df: Data frame holding DLC body part data.
    :param np.ndarray label: Label from B-SOID applied to DLC data.
    :param list bodyparts: Body parts to render.
    :param str savedir: Directory to where figures can be saved.
    :param str savename: Name of the saved figure.
    :param tuple length_limits: In form (low, high), indicates what range 
    of bout lengths should be used for visualization. A None at either 
    position indicates no limit on the upper / lower bound. 
    Defaults to (None, None), no restriction.
    :param int locomotion_label: Specifies which labels corresponds to 
    the locomotion group. Defaults to LOCOMOTION_LABELS.
    :param int plot_N_runs: The number of runs we plot. Defaults to all.
    :param float threshold: Threshold separating locomotion movement from 
    paw rest, defaults to MOVEMENT_THRESHOLD
    :param bool save_figure: Whether to save the figure, defaults to True
    :param bool show_figure: Whether to show the figure, defaults to True
    """
    
    starts, ends, _ = select_N_locomotion_sequences(
        label=label, N=plot_N_runs, locomotion_labels=locomotion_label, 
        length_limits=length_limits
        )
    
    colorclass = plt.cm.ScalarMappable(cmap="rainbow")
    C = colorclass.to_rgba(np.linspace(0, 1, len(bodyparts)))
    colors = C[:, :3]

    df, _ = filter_nonpawrest_motion(df=df, label=label, show_nonpaw=True, 
                                     threshold=threshold, locomotion_labels=locomotion_label, 
                                     average_pawrest=True)
    
    # we will render as figure each sequence of movement into a 2D grid
    for start, end in zip(starts, ends):
        _, ax = plt.subplots()

        for bpt_idx, bpt in enumerate(bodyparts):
            x, y = df[bpt, 'x'].to_numpy(), df[bpt, 'y'].to_numpy()
            run_x, run_y = x[start:end+1], y[start:end+1]

            if 'paw' in bpt:
                ax.scatter(run_x, run_y, marker='o', color=colors[bpt_idx], label=bpt)
            else:
                ax.plot(run_x, run_y, color=colors[bpt_idx], label=bpt)
        ax.set_xlabel('X (pixel)')
        ax.set_ylabel('Y (pixel)')
        ax.set_xlim([0, 1100]); ax.set_ylim([0, 1100])
        ax.legend()

        plt.title(f"Locomotion Paw Stationary Moments {start}~{end} \n(Threshold={threshold}, Length Limits={length_limits})")

        if save_figure:
            plt.savefig(os.path.join(
                savedir, 
                f"{savename.replace('.png', '')}_{start}To{end}_thresh{threshold}_lim{length_limits[0]}To{length_limits[1]}.png")
                )
        if show_figure:
            plt.show()
        else:
            plt.close()

def visualize_locomotion_stats(label : np.ndarray, 
                               figure_name : str,
                               save_path : str,
                               locomotion_label : list,
                               num_bins : int=20,
                               interval : int=40*60*5,
                               use_logscale : bool=True,
                               save_figure : bool=True,
                               show_figure : bool=True):
    """
    Visualizes some stats about locomotion labels, namely:
    - how many locomotion bouts there are
    - how many of each length of locomotion bouts there are
    - distribution of the distance between consecutive locomotion bouts
    - distribution of locomotion over time for every 5 min 

    :param np.ndarray label: Labels given to the time series of DLC data.
    :param str figure_name: Name of the saved figure file.
    :param str save_path: Path to which the figure is saved if needed.
    :param List[int] locomotion_label: List of labels corresponding to locomotion.
    :param int num_bins: The number of bins in the histograms, defaults to 20.
    :param int interval: Size of interval in number of frames we want to consider 
    in showing the distribution of locomotion over time. Defaults to 40*60*5, 
    5 min interval with fps=40.
    :param bool use_logscale: Whether to use log scale for the y axis SPECIFICALLY 
    for the figure of bout length between bouts, defaults to True.
    :param bool save_figure: Whether to save the figure, defaults to True
    :param bool show_figure: Whether to show the figure, defaults to True
    """
    # 1) Extract all locomotion bouts
    locomotion_idx, locomotion_lengths = find_locomotion_sequences(
        label=label,
        locomotion_label=locomotion_label,
        length_limits=(None, None)
        )

    # 2) Visualize info:
    # - How many locomotion bouts there are
    # - How many of each length of locomotion bouts there are
    # - What's the distribution of the distance between two locomotion bouts (curious if bouts are broken down but close together)
    # - What's the distribution of locomotion for every 5 min interval (just curious)
    _, axes = plt.subplots(3, 1)
    axes[0].hist(locomotion_lengths, bins=num_bins, color="blue")
    axes[0].set_title("Locomotion Bout Lengths")
    if use_logscale: axes[0].set_yscale('log')
    non_loc_bout_start = locomotion_idx[:-1] + locomotion_lengths[:-1]
    non_loc_bout_end = locomotion_idx[1:] - 1
    axes[1].hist(non_loc_bout_end - non_loc_bout_start, bins=num_bins, color="red")
    axes[1].set_title(f"In-between of Locomotion Bout Lengths {'(log scale)' if use_logscale else ''}")
    if use_logscale: axes[1].set_yscale('log')
    # plot number of bouts frequency per interval
    interval_start, interval_end = 0, interval - 1
    interval_locomotion_quantity = []
    while interval_start < len(label):
        interval_label = label[interval_start:interval_end+1]
        interval_locomotion_quantity.append(np.sum(np.isin(interval_label, locomotion_label)))
        interval_start += interval; interval_end += interval
    axes[2].plot(interval_locomotion_quantity, color="green")
    axes[2].set_title(f"Number of Locomotion Frame Per Interval ({interval} frames)")
    
    plt.suptitle(f"Locomotion Stats ({len(locomotion_idx)} bouts)")
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(os.path.join(save_path, figure_name))

    if show_figure: plt.show()
    else: plt.close()

def visualize_stepsize_in_locomotion_in_single_mouse(
        stepsize_yaml : str,
        title : str,
        savedir : str,
        savename : str,
        binsize : int=None,
        show_figure : bool=True,
        save_figure : bool=True
        ):
    """
    Visualizes the histogram of step sizes in mice in locomotion, 
    focusing one mouse at a time and rendering the results for each 
    of right fore paw, left fore paw, right hind paw, left hind paw.

    :param str stepsize_yaml: Path to yaml holding data for one mouse.
    :param str title: Title of the rendered figure.
    :param str savedir: Directory to which the figure would be saved.
    :param str savename: File name of the figure to be saved.
    :param int binsize: Integer indicating bin size shared among histograms. 
    Defaults to not being specified.
    :param bool show_figure: Whether to show the figure, defaults to True
    :param bool save_figure: Whether to save the figure, defaults to True
    """
    POSITION = [[0,0], [0,1], [1,0], [1,1]]

    _, axes = plt.subplots(2, 2, figsize=(10,5))

    min_stepsize, max_stepsize = float("inf"), float("-inf")

    for bpt, position in zip(
        ['rightforepaw', 'leftforepaw', 'righthindpaw', 'lefthindpaw'], 
        POSITION
        ):
        ax = axes[position[0], position[1]]
        extrema = visualize_stepsize_in_locomotion(stepsize_yaml=stepsize_yaml,
                                                   bodyparts=[bpt],
                                                   title="", 
                                                   binsize=binsize,
                                                   ax=ax)
        if len(extrema) != 0:
            min_stepsize = min(min_stepsize, extrema[bpt]["min"])
            max_stepsize = max(max_stepsize, extrema[bpt]["max"])
    
    if len(extrema) != 0:
        for pos in POSITION:
            ax = axes[pos[0], pos[1]]
            ax.set_xlim(min_stepsize, max_stepsize)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(os.path.join(savedir, savename))
    
    if show_figure:
        plt.show()
    else:
        plt.close()

def visualize_stepsize_in_locomotion_in_multiple_mice(
        yamls : list,
        title : str,
        savedir : str,
        savename : str,
        binsize : int=None,
        show_figure : bool=True,
        save_figure : bool=True
        ):
    """
    Visualizes the histogram of step sizes in mice in locomotion, 
    focusing on multiple mice for the same body part. Each of right 
    fore paw, left fore paw, right hind paw, left hind paw will be rendered
    at once for all mice, in separate graphs on the same figure.

    :param list yamls: A list of paths to yaml holding data for one mouse.
    :param str title: Title of the rendered figure.
    :param str savedir: Directory to which the figure would be saved.
    :param str savename: File name of the figure to be saved.
    :param int binsize: Integer indicating bin size shared among histograms. 
    Defaults to not being specified.
    :param bool show_figure: Whether to show the figure, defaults to True
    :param bool save_figure: Whether to save the figure, defaults to True
    """
    if len(yamls) == 0:
        return 
    # we arrange the mice into a grid closest to a square as possible
    num_mouse = int(len(yamls))
    num_rows  = int(sqrt(num_mouse) // 1)
    num_cols  = int((num_mouse - 1) // num_rows + 1)

    for bpt in ['rightforepaw', 'leftforepaw', 'righthindpaw', 'lefthindpaw']:
        _, axes = plt.subplots(num_rows, num_cols, figsize=(15, 7))

        min_stepsize, max_stepsize = float("inf"), float("-inf")

        for mouse_idx, yaml in enumerate(yamls):
            row_idx, col_idx = int(mouse_idx % num_rows), int(mouse_idx // num_rows)

            ax = axes[row_idx, col_idx]
            extrema = visualize_stepsize_in_locomotion(stepsize_yaml=yaml,
                                                       bodyparts=[bpt],
                                                       title=get_mousename(yaml), 
                                                       binsize=binsize,
                                                       ax=ax)
            if len(extrema) != 0:
                min_stepsize = min(min_stepsize, extrema[bpt]["min"])
                max_stepsize = max(max_stepsize, extrema[bpt]["max"])
    
        for mouse_idx in range(num_mouse):
            row_idx, col_idx = mouse_idx % num_rows, mouse_idx // num_rows
            ax = axes[row_idx, col_idx]
            ax.set_xlim(min_stepsize, max_stepsize)
            if row_idx == (num_rows - 1):
                ax.set_xlabel('Step Size (pixel)')
            if col_idx == (num_cols - 1):
                ax.set_ylabel('Frequency')
        
        plt.suptitle(f"{title} ({bpt})")
        plt.tight_layout()
    
        if save_figure:
            plt.savefig(os.path.join(savedir, f'{savename}_{bpt}'))
        
        if show_figure:
            plt.show()
        else:
            plt.close()

def visualize_stepsize_in_locomotion_in_mice_groups(
        yamls_groups : list,
        groupnames : list,
        title : str,
        savedir : str,
        savename : str,
        binsize : int=None,
        show_figure : bool=True,
        save_figure : bool=True
        ):
    """
    Visualizes the histogram of step sizes in mice in locomotion, 
    focusing on multiple mice for the same body part. Each of right 
    fore paw, left fore paw, right hind paw, left hind paw will be rendered
    at once for all mice, where all step sizes are put together.

    :param list yamls_groups: A list of 'list of paths' to yaml holding data for one mouse.
    Each list is a group of yamls.
    :param list groupnames: A list of the name of groups of yamls corresponding to 
    each group given as 'yaml_groups'.
    :param str title: Title of the rendered figure.
    :param str savedir: Directory to which the figure would be saved.
    :param str savename: File name of the figure to be saved.
    :param int binsize: Integer indicating bin size shared among histograms. 
    Defaults to not being specified.
    :param bool show_figure: Whether to show the figure, defaults to True
    :param bool save_figure: Whether to save the figure, defaults to True
    """
    # for each body part in question, we create a separate figure
    for bpt, bpt_title in zip(
        ['rightforepaw',   'leftforepaw',   'righthindpaw',   'lefthindpaw'  ], 
        ['Right Fore Paw', 'Left Fore Paw', 'Right Hind Paw', 'Left Hind Paw']
        ):
        _, axes = plt.subplots(len(groupnames), 1)
        min_stepsize, max_stepsize = float("inf"), float("-inf")

        # for each mouse group
        for group_idx, (yamls, groupname) in enumerate(zip(yamls_groups, groupnames)):
            ax = axes if (len(groupnames) == 1) else axes[group_idx]

            # create a yaml that merges all such yamls together
            yaml_contents = [read_stepsize_yaml(yml) for yml in yamls]
            merged_yaml = {}
            
            for yaml_idx, content in enumerate(yaml_contents):
                for key, val in content.items():
                    merged_yaml[f'{key}_{yaml_idx}'] = val
                
            extrema = visualize_stepsize_in_locomotion(stepsize_yaml=merged_yaml,
                                                       bodyparts=[bpt],
                                                       title=groupname, 
                                                       binsize=binsize,
                                                       ax=ax)
            if len(extrema) != 0:
                min_stepsize = min(min_stepsize, extrema[bpt]["min"])
                max_stepsize = max(max_stepsize, extrema[bpt]["max"])
            
            ax.set_title(groupname)
            ax.set_xlabel('Step Size (pixel)')
            if group_idx == 0:
                ax.set_ylabel('Frequency')
        
        for i in range(len(groupnames)):
            ax = axes if (len(groupnames) == 1) else axes[i]
            ax.set_xlim(min_stepsize, max_stepsize)
                
        plt.suptitle(f"{title} - {bpt_title}")
        plt.tight_layout()
    
        if save_figure:
            plt.savefig(os.path.join(savedir, f'{savename}_{bpt}'))
        
        if show_figure:
            plt.show()
        else:
            plt.close()

def visualize_stepsize_in_locomotion(stepsize_yaml, #str or dict,
                                     bodyparts : list,
                                     title : str,
                                     binsize : int=None,
                                     ax : Axes=None):
    extrema = {}

    if type(stepsize_yaml) == dict:
        content = stepsize_yaml
    elif type(stepsize_yaml) == str:
        content = read_stepsize_yaml(stepsize_yaml)

    if ax is None:
        plt.title(f"{title}")
        plt.xlabel("Step Size (pixel)")
        plt.ylabel("Frequency")
        plt.show()
    else:
        ax.set_title(f"{title}")

    # deal with the (rare) case where there is no locomotion entry
    if len(content) == 0:
        extrema = {}
        return extrema

    # for every body part except the "end" entry which isn't    
    for bpt in [bpt for bpt in list(content.values())[0] 
                if (bpt != 'end' and bpt in bodyparts)]:
        all_stepsizes = []

        for elem in content.values():
            new_stepsize = [stepsize for stepsize in elem[bpt]['diff'] 
                            if stepsize > CUTOFF]
            all_stepsizes.extend(new_stepsize)
        
        # visualize the result into a histogram
        bins = range(0, (int(max(all_stepsizes)) // binsize + 1) * binsize, binsize)
        if ax is None:
            plt.hist(all_stepsizes, bins=bins)
        else:
            ax.hist(all_stepsizes, bins=bins)
    
        extrema[bpt] = {"min" : min(all_stepsizes), "max" : max(all_stepsizes)}
    return extrema

# HELPER

def find_locomotion_sequences(label : np.ndarray,
                              locomotion_label : list=LOCOMOTION_LABELS,
                              length_limits : tuple=(None, None)):
    """
    Find the set of locomotion sequences as continuous labels of any label 
    in 'locomotion_label', found within label.
    Returns those starting indices & lengths for such sequences, if they 
    fall within the given limits in length.

    :param np.ndarray label: Label used to extract sequences of locomotion.
    :param list locomotion_label: Label signifying locomotion, defaults to LOCOMOTION_LABELS.
    :param tuple length_limits: Lower & upper limit in frame for locomotion sequence we 
    consider, defaults to any length.
    :return np.ndarray locomotion_idx: Index of the start of locomotion sequences.
    :return np.ndarray locomotion_lengths: Lengths of the locomotion sequences.
    """
    len_lowlim, len_highlim = process_upper_and_lower_limit(length_limits)

    # identify which of locomotion_label exist in label
    contained_loc_lbl = np.array(locomotion_label)[np.isin(locomotion_label, np.unique(label))]
    # merge all instances of locomotion labels into one label (first in list of unique labels)
    for loc_lbl in contained_loc_lbl:
        if loc_lbl != contained_loc_lbl[0]:
            label[label == loc_lbl] = contained_loc_lbl[0]

    # obtain all locomotion-labeled sequences from labels
    run_values, run_start, run_lengths = find_runs(label)
    locomotion_within_array = np.logical_and(
        np.isin(run_values, locomotion_label), # is locomotion in label
        np.logical_and(len_lowlim <= run_lengths, run_lengths <= len_highlim)
        ) # is within allowed run range
    locomotion_idx = run_start[locomotion_within_array]
    locomotion_lengths = run_lengths[locomotion_within_array]

    return locomotion_idx, locomotion_lengths

def find_paw_movements(movement : np.ndarray, threshold : float=MOVEMENT_THRESHOLD):
    """
    Finds paw movements based on looking at values above threshold.
    :returns np.ndarray move_start, move_end, move_middle:
    """
    paw_movements = (movement > threshold).tolist()
    run_values, run_starts, run_lengths = find_runs(paw_movements)
    # find start, end then middle of paw movements
    moves = (run_values == True)
    move_start = run_starts[moves]
    move_end = run_starts[moves] + run_lengths[moves] - 1
    move_middle = (move_start + move_end) // 2
    return move_start, move_end, move_middle

def filter_nonpawrest_motion(df : pd.DataFrame, 
                             label : np.ndarray,
                             show_nonpaw : bool=False,
                             threshold : float=MOVEMENT_THRESHOLD,
                             locomotion_labels : list=LOCOMOTION_LABELS, 
                             average_pawrest : bool=True):
    """
    Takes in a data frame holding DLC data as well as an array of BSOID label
    for it, filtering out:
    - any non-locomotion label frame 
    - any locomotion label frame which is followed by movement above 'threshold'
    If average_pawrest, we take each paw rest, averaging the time while it is resting.

    :param pd.DataFrame df: DataFrame holding DLC data.
    :param np.ndarray label: BSOID label for the DLC data. 
    :param bool show_nonpaw: Whether to show non-paw body parts, 
    defaults to False
    :param float threshold: Threshold separating locomotion movement from 
    paw rest, defaults to MOVEMENT_THRESHOLD
    :param list locomotion_labels: Integer labels corresponding to locomotion groups, 
    defaults to LOCOMOTION_LABEL (38, for YAC128 network).    
    :param bool average_pawrest: Whether to make each continuous paw rest that is identified
    averaged, so that a single point is identified. E.g. if the X coord is: 
    [NaN, 3, 3.2, 3.1, NaN, 5, 4.8, NaN]..., we average this to be:
    [NaN, 3.1, 3.1, 3.1, NaN, 4.9, 4.9, NaN].
    """
    # remove entries in df where movement of 'paws' is above threshold
    unique_bpts = np.unique(df.columns.get_level_values('bodyparts'))
    for bpt in unique_bpts:
        if 'paw' in bpt:
            x, y = df[bpt, 'x'].to_numpy(), df[bpt, 'y'].to_numpy()
            movement = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            movement = np.insert(movement, 0, 0) # pad first position as no movement
            df.loc[movement > threshold, [bpt]] = np.nan
        # completely remove entries for parts like snout and tailbase
        else:
            if not show_nonpaw:
                df.loc[:, [bpt]] = np.nan
    # remove entries in df that aren't locomotion bouts
    df.loc[label != locomotion_labels[0], :] = np.nan
    
    # if average_pawrest: 1) identify paw rest sequences
    # 2) replace the entire sequence with its average
    if average_pawrest:
        # we also return an "averaged" data frame which holds, for every 
        # paw sequence: body part, first time stamp, length, x average, y average
        average_array = []
        for bpt in unique_bpts:
            if 'paw' in bpt:
                isnan_x = np.isnan(df[bpt, 'x'].to_numpy())
                run_values, run_starts, run_lengths = find_runs(isnan_x)
                
                for nonnan_idx in np.where(run_values == False)[0]:
                    start = run_starts[nonnan_idx].item()
                    length = run_lengths[nonnan_idx].item()
                    end = start + length - 1
                    # get the average of x and y
                    avg_x = np.mean(df.loc[start:end, (bpt, 'x')])
                    avg_y = np.mean(df.loc[start:end, (bpt, 'y')])

                    # change the original data frame
                    df.loc[start:end, (bpt, 'x')] = avg_x
                    df.loc[start:end, (bpt, 'y')] = avg_y

                    # add a new row to the new dataframe
                    new_row = (bpt, start, length, avg_x, avg_y)
                    average_array.append(new_row)
        # make a dataframe out of it
        columns = [COL_BODYPART, COL_START, COL_LENGTH, COL_X_AVG, COL_Y_AVG]
        average_df = pd.DataFrame(data=average_array, 
                                  columns=columns)

    else:
        average_df = None
    return df, average_df

def select_N_locomotion_sequences(label : np.ndarray,
                                  N : int=float("inf"),
                                  locomotion_labels : list=LOCOMOTION_LABELS, 
                                  length_limits : tuple=(None,None)):
    """
    From label, selects N locomotions sequences that match with locomotion_labels 
    and are within the specified length_limits.

    :param np.ndarray label: BSOID label for the DLC data. 
    :param int N: The number of sequences we extract. Defaults to all existing ones.
    :param list locomotion_labels: Integer labels corresponding to locomotion groups, 
    defaults to LOCOMOTION_LABEL (38, for YAC128 network).    
    :param tuple length_limits: Lower & upper limits in length for locomotion
    snippets to use to generate videos, defaults to no restriction.
    
    :return list starts: A list of start for the sequences. 
    :return list ends: A list of end for the sequences. Same order as starts.
    :return list lengths: A list of the lengths for the sequences. Same order as 
    starts.
    """
    loc_starts, loc_lengths = find_locomotion_sequences(
        label=label, locomotion_label=locomotion_labels, length_limits=length_limits
        )
    # generate videos for 'num_runs' locomotion bout
    starts, ends, lengths = [], [], []
    
    N = min(N, len(loc_starts))
    for i in range(N):
        start, length = loc_starts[i], loc_lengths[i]
        end = start + length - 1
        starts.append(start); ends.append(end); lengths.append(length)
    
    return starts, ends, lengths

def read_stepsize_yaml(stepsize_yaml : str):
    """
    Read a yaml file holding "step size" information
    for locomotion.

    :param str stepsize_yaml: Path to a yaml holding step size info.
    :return Dict: A dictionary holding data for step size. Keys are the
    starting frame number, with entries: 
    [end, rightforepaw, leftforepaw, righthindpaw, lefthindpaw].
    """
    with open(stepsize_yaml, 'r') as f:
        yaml_content = f.read()
    content = yaml.safe_load(yaml_content)
    return content