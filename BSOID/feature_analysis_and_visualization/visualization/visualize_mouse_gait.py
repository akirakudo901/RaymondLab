# Author: Akira Kudo
# Created: 2024/03/31
# Last updated: 2024/06/17

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils import find_runs, process_upper_and_lower_limit

LOCOMOTION_LABELS = [38]
MOVEMENT_THRESHOLD = 0.75

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