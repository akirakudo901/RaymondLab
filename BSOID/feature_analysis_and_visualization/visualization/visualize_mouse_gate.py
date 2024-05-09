# Author: Akira Kudo
# Created: 2024/03/31
# Last updated: 2024/05/08

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_analysis_and_visualization.utils import find_runs
from feature_analysis_and_visualization.utils import process_upper_and_lower_limit

LOCOMOTION_LABELS = [38]
MOVEMENT_THRESHOLD = 5

def visualize_mouse_gate(df : pd.DataFrame, 
                         label : np.ndarray, 
                         bodyparts : list, 
                         length_limits : tuple=(None, None),
                         locomotion_label : list=LOCOMOTION_LABELS, 
                         plot_N_runs : int=float("inf")):
    """
    Visualizes the mouse gate by taking in data & label, 
    identifying sequences of gates in the data, and then
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
    len_lowlim, len_highlim = process_upper_and_lower_limit(length_limits)

    # obtain all locomotion-labeled sequences from labels
    run_values, run_start, run_lengths = find_runs(label)
    locomotion_within_array = np.logical_and(
        np.isin(run_values, locomotion_label), # is locomotion in label
        np.logical_and(len_lowlim <= run_lengths, run_lengths <= len_highlim)
        ) # is within allowed run range
    locomotion_idx = run_start[locomotion_within_array]
    locomotion_lengths = run_lengths[locomotion_within_array]
    
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