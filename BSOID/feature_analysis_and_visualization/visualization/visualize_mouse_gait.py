# Author: Akira Kudo
# Created: 2024/03/31
# Last updated: 2024/07/29

import os

from math import sqrt
from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
import yaml

from ..utils import find_runs, get_mousename, process_upper_and_lower_limit, BPT_ABBREV
from ..analysis import analyze_mouse_gait

LOCOMOTION_LABELS = [38]
MOVEMENT_THRESHOLD = 0.75
STEPSIZE_MIN = 7 # a cutoff for a stepsize that is considered the same position

# column names for an averaged paw movement data frame
COL_BODYPART = 'bodypart'
COL_START = 'start'
COL_LENGTH = 'length'
COL_X_AVG = 'x_avg'
COL_Y_AVG = 'y_avg'

MEAN_LINESTYLE = "--"

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

def visualize_mouse_gait_speed_of_specific_sequences(
        df : pd.DataFrame,
        sequences : list,
        bodyparts : list,
        paw_rest_color : str,
        savedir : str,
        save_prefix : str,
        threshold : float=MOVEMENT_THRESHOLD,
        save_figure : bool=True,
        show_figure : bool=True
        ):
    """
    Visualizes speed of the mouse gait by taking in data and a set
    of start & end of locomotion sequences. Each body part specified
    will be a single subplot, stacked vertically into a single figure.
    Also show any start of paw rest as vertical line.

    :param pd.DataFrame df: Dataframe holding x, y data of bodyparts.
    :param list sequences: A list of tuples, each of two integers indicating
    the start and end of a locomotion bout.
    :param list bodyparts: A list of body part names to visualize for.
    :param str paw_rest_color: Color to indicate frames of paw rests.
    :param str savedir: Directory to where figures can be saved.
    :param str savename: Name of the saved figure.
    :param float threshold: Threshold separating locomotion movement from 
    paw rest, defaults to MOVEMENT_THRESHOLD
    :param bool save_figure: Whether to save the figure, defaults to True
    :param bool show_figure: Whether to show the figure, defaults to True
    """
    bpt_in_df = df.columns.get_level_values("bodyparts")
    bodyparts = np.array(bodyparts)[np.isin(bodyparts, bpt_in_df)]

    for start, end in sequences:
        _, axes = plt.subplots(nrows=len(bodyparts), ncols=1)

        max_speed = float("-inf")

        for bpt_idx, bpt in enumerate(bodyparts):
            ax = axes if len(bpt) == 1 else axes[bpt_idx]
            # want to show: the speed in a graph
            # also add paw rest starts in horizontal bars of colors
            x, y = df[(bpt, 'x')].to_numpy(), df[(bpt, 'y')].to_numpy()

            run_x, run_y = x[start:end+1], y[start:end+1]
            run_speed = np.sqrt(np.diff(run_x)**2 + np.diff(run_y)**2)
            max_speed = np.max([max_speed, np.max(run_speed).item()])

            # finding paw rests v1: look at anything below threshold
            paw_rests = np.nonzero(run_speed <= threshold)[0] + start
            # finding paw rests v2: look at anything below threshold
            # where the speed also decreased by a certain amount
            ax.bar(paw_rests, max_speed, width=1, color=paw_rest_color)

            # plot the line graph on top of colored bars
            ax.plot(range(start, end), run_speed)
            
            ax.set_title(f'{bpt}')
            
        # adjust the maximum displayed Y axis value to the highest of all body parts
        # such that their vertical height mean the same
        max_speed = max_speed * 1.1
        for ax_idx in range(len(bodyparts)):
            ax = axes if (len(bodyparts) == 1) else axes[ax_idx]
            ax.set_ylim(top=max_speed)
        
        plt.suptitle("Body Parts Speed & Paw Rest Frames" + 
                     f"\nSequence {start}~{end}, Threshold={threshold}")
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(os.path.join(savedir, 
                                     f"{save_prefix}_{bpt}_{start}To{end}_GaitSpeed"))

        if show_figure: plt.show()
        else: plt.close()
        

def visualize_mouse_paw_rests_in_locomomotion(
        df : pd.DataFrame,
        label : np.ndarray, 
        bodyparts : list,
        savedir : str,
        savename : str,
        also_render_in_line : list,
        averaged : bool=True,
        annotate_framenum : bool=False,
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
    Renders such paw rests as well as the trajectory of specified body parts as a 
    single figure per such sequence. Which body part are rendered can be specified.

    :param pd.DataFrame df: Data frame holding DLC body part data.
    :param np.ndarray label: Label from B-SOID applied to DLC data.
    :param list bodyparts: Body parts to render. Any body part with 'paw' in it
    will be depicted in paw rests, and any others in lines of the trajectory.
    :param str savedir: Directory to where figures can be saved.
    :param str savename: Name of the saved figure.
    :param list also_render_in_line: A list of body part names in string which will be 
    rendered as lines on top of the individual paws being depicted.
    :param bool averaged: Create figures which paw positions are averaged
    per each identified paw rest, defaults to True
    :param bool annotate_framenum: Whether to annotate each paw identified
    with a frame number for the start of the paw. Defaults to False.
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

    filt_df, avg_df = filter_nonpawrest_motion(df=df, label=label, show_nonpaw=True, 
                                               threshold=threshold, locomotion_labels=locomotion_label, 
                                               average_pawrest=True, ignore_close_paw=STEPSIZE_MIN)
    
    # we will render as figure each sequence of movement into a 2D grid
    for start, end in zip(starts, ends):
        _, ax = plt.subplots()

        for bpt_idx, bpt in enumerate(bodyparts):

            # render the trajectory
            if (bpt in also_render_in_line) or ("paw" not in bpt):
                x, y = df[bpt, 'x'].to_numpy(), df[bpt, 'y'].to_numpy()
                run_x, run_y = x[start:end+1], y[start:end+1]
                ax.plot(run_x, run_y, color=colors[bpt_idx], label=bpt)

            # render the paw rests
            if 'paw' in bpt:
                if averaged:
                    bout = avg_df[(avg_df[COL_START] > start) & 
                                  (avg_df[COL_START] < end) & 
                                  (avg_df[COL_BODYPART] == bpt)]
                    run_x, run_y = bout[COL_X_AVG].to_numpy(), bout[COL_Y_AVG].to_numpy()
                    run_start = bout[COL_START].to_numpy()
                else:
                    x, y = filt_df[bpt, 'x'].to_numpy(), filt_df[bpt, 'y'].to_numpy()
                    run_x, run_y = x[start:end+1], y[start:end+1]
                    run_start = np.nonzero(~np.isnan(x))[0]
                    run_start = run_start[(run_start >= start) & (run_start <= end)]
                
                ax.scatter(run_x, run_y, marker='3', color=colors[bpt_idx], label=bpt)

                # if specified, annotate each paw rest with their starting frame number
                if annotate_framenum:
                    offset = 3
                    for x_coord, y_coord, rest_start in zip(run_x, run_y, run_start):
                        ax.annotate(rest_start.item(), 
                                    (x_coord.item()+offset, y_coord.item()+offset),
                                    fontsize="xx-small")

        ax.set_xlabel('X (pixel)')
        ax.set_ylabel('Y (pixel)')
        ax.set_xlim([0, 1100]); ax.set_ylim([0, 1100])
        ax.legend()

        title = (f"Locomotion Paw Stationary Moments {start}~{end} \n" +
                 f"(Threshold={threshold}, Length Limits={length_limits}, Averaged={averaged})")
        if len(also_render_in_line) > 0:
            title += f"\nTrajectory shown for {','.join(also_render_in_line)}"
        plt.title(title)
        
        if save_figure:
            render_in_line_namepart = ('_' + '_'.join([BPT_ABBREV[bpt] for bpt in also_render_in_line])) \
                                      if len(also_render_in_line) > 0 else ''
            new_savename = (f"{savename.replace('.png', '')}_" + 
                            f"{start}To{end}_thresh{threshold}_lim{length_limits[0]}To{length_limits[1]}" + 
                            f"{'_annotated' if annotate_framenum else ''}{render_in_line_namepart}.png")
            plt.savefig(os.path.join(savedir, new_savename))
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
        stepsize_info, # str or dict
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

    :param str or dict stepsize_info: Info on the step sizes. Can either 
    be a full path to a yaml holding step size info, or a dictionary of a
    specific format.
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
        extrema = visualize_stepsize_in_locomotion(stepsize_info=stepsize_info,
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
        stepsizes : list, # list of str or dict
        mousenames : list,
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

    :param list stepsizes: A list of either paths to yaml holding data for one mouse, 
    or dicts holding data for one mouse.
    :param list mousenames: A list of mouse names matching the step size entries, only
    required when dictionaries are passed. When paths to yamls are passed and mousenames
    is None, we can infer the names from paths.
    :param str title: Title of the rendered figure.
    :param str savedir: Directory to which the figure would be saved.
    :param str savename: File name of the figure to be saved.
    :param int binsize: Integer indicating bin size shared among histograms. 
    Defaults to not being specified.
    :param bool show_figure: Whether to show the figure, defaults to True
    :param bool save_figure: Whether to save the figure, defaults to True
    """
    if len(stepsizes) == 0:
        return 
    if type(stepsizes[0]) == dict and mousenames is None:
        raise Exception("When stepsizes is a list of dict, mousenames should be given...")
    if mousenames is not None and len(stepsizes) != len(mousenames):
        raise Exception("Length of stepsizes and mousenames should match if given, but are " + 
                        f"instead respectively: {len(stepsizes)} and {len(mousenames)}...")

    # we arrange the mice into a grid closest to a square as possible
    num_mouse = int(len(stepsizes))
    num_rows  = int(sqrt(num_mouse) // 1)
    num_cols  = int((num_mouse - 1) // num_rows + 1)

    for bpt in ['rightforepaw', 'leftforepaw', 'righthindpaw', 'lefthindpaw']:
        _, axes = plt.subplots(num_rows, num_cols, figsize=(15, 7))

        min_stepsize, max_stepsize = float("inf"), float("-inf")

        for mouse_idx, stepsize_info in enumerate(stepsizes):
            row_idx, col_idx = int(mouse_idx % num_rows), int(mouse_idx // num_rows)
            mname = mousenames[mouse_idx] if (mousenames is not None) else get_mousename(stepsize_info)

            ax = axes[row_idx, col_idx]
            extrema = visualize_stepsize_in_locomotion(stepsize_info=stepsize_info,
                                                       bodyparts=[bpt],
                                                       title=mname, 
                                                       binsize=binsize,
                                                       ax=ax,
                                                       show_num_bouts=True)
            if len(extrema) != 0:
                min_stepsize = min(min_stepsize, extrema[bpt]["min"])
                max_stepsize = max(max_stepsize, extrema[bpt]["max"])
    
        for mouse_idx in range(num_mouse):
            row_idx, col_idx = int(mouse_idx % num_rows), int(mouse_idx // num_rows)
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
        stepsize_groups : list, # list of list of (str or dict)
        groupnames : list,
        title : str,
        savedir : str,
        savename : str,
        group_colors : list=None,
        binsize : int=None,
        show_figure : bool=True,
        save_figure : bool=True
        ):
    """
    Visualizes the histogram of step sizes in mice in locomotion, 
    focusing on multiple mice for the same body part. Each of right 
    fore paw, left fore paw, right hind paw, left hind paw will be rendered
    at once for all mice, where all step sizes are put together.

    :param list stepsize_groups: A list of lists, each list being a group treated together.
    Each group is a list of either paths to yaml holding data for single mice, or dicts.
    :param list groupnames: A list of the name of groups of yamls corresponding to 
    each group given as 'yaml_groups'.
    :param str title: Title of the rendered figure.
    :param str savedir: Directory to which the figure would be saved.
    :param str savename: File name of the figure to be saved.
    :param list group_colors: A list of strings specifying the color of each group. 
    Defautls to being all blue.
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

        # if group colors isn't specified, default to being all blue
        if group_colors is None:
            group_colors = ["blue"] * len(groupname)

        # for each mouse group
        for group_idx, (stepsizes, groupname, groupcolor) in enumerate(
                zip(stepsize_groups, groupnames, group_colors)):
            ax = axes if (len(groupnames) == 1) else axes[group_idx]

            # create a yaml that merges all such yamls together
            if type(stepsizes[0]) == str:
                yaml_contents = [read_stepsize_yaml(yml) for yml in stepsizes]
            else:
                yaml_contents = stepsizes
            merged_yaml = {}
            
            for yaml_idx, content in enumerate(yaml_contents):
                for key, val in content.items():
                    merged_yaml[f'{key}_{yaml_idx}'] = val
                
            extrema = visualize_stepsize_in_locomotion(stepsize_info=merged_yaml,
                                                       bodyparts=[bpt],
                                                       title=groupname,
                                                       color=groupcolor, 
                                                       binsize=binsize,
                                                       ax=ax,
                                                       show_num_bouts=True,
                                                       show_mean=True)
            if len(extrema) != 0:
                min_stepsize = min(min_stepsize, extrema[bpt]["min"])
                max_stepsize = max(max_stepsize, extrema[bpt]["max"])
            
            ax.set_title(groupname)
            if group_idx == (len(groupnames) - 1):
                ax.set_xlabel('Step Size (pixel)')
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

def visualize_stepsize_in_locomotion(stepsize_info, #str or dict,
                                     bodyparts : list,
                                     title : str,
                                     color : str="blue",
                                     binsize : int=None,
                                     ax : Axes=None,
                                     show_num_bouts : bool=False,
                                     show_mean : bool=False):
    """
    Visualizes the frequency of step sizes for the specified body parts
    in the given information on step sizes. A histogram is created and
    shown either on its own or in on the given matplotlib axis object.

    :param str or dict stepsize_info: Info on the step sizes. Can either 
    be a full path to a yaml holding step size info, or a dictionary of a
    specific format.
    :param list bodyparts: A list of body parts we visualize.
    :param str title: Title of the figure.
    :param str color: Color of the histogram, defaults to blue.
    :param int binsize: Width of a bin, defaults to auto determined.
    :param Axes ax: Optionally an axis to render the histogram on, 
    if not given, a new figure is created.
    :param bool show_num_bouts: Whether to indicate the number of bouts 
    and strides used for each mouse, shown as part of the title of each graph / axis. 
    Defaults to False.
    :param bool show_mean: Whether to show the mean of the given step size info
    as a vertical line. Defaults to False. 
   
    :return dict extrema: A dictionary mapping each body part name with 
    entries of 'max' and 'min', specifying each the maximum and minimum 
    step sizes for that body part.
    """
    MEANLINE_COLOR = "black"
    extrema = {}

    if type(stepsize_info) == dict:
        content = stepsize_info
    elif type(stepsize_info) == str:
        content = read_stepsize_yaml(stepsize_info)
        
    if ax is None:
        plt.xlabel("Step Size (pixel)")
        plt.ylabel("Frequency")

    # deal with the case where there is no locomotion entry
    if len(content) == 0:
        if ax is None:
            plt.title(title)
        else:
            ax.set_title(title)
        
        if show_num_bouts:
            title = f"{title} ({len(content)} bouts)"
        
        extrema = {}
        return extrema

    # for every body part except the "end" entry which isn't    
    for bpt in [bpt for bpt in list(content.values())[0] 
                if (bpt != 'end' and bpt in bodyparts)]:
        all_stepsizes = []

        for elem in content.values():
            new_stepsize = [stepsize for stepsize in elem[bpt]['diff'] 
                            if stepsize > STEPSIZE_MIN]
            all_stepsizes.extend(new_stepsize)

        if show_num_bouts:
            title = f"{title} ({len(content)} bouts {len(all_stepsizes)} strides)"
        
        # visualize the result into a histogram
        bins = range(0, (int(max(all_stepsizes)) // binsize + 1) * binsize, binsize)
        if ax is None:
            plt.hist(all_stepsizes, bins=bins, color=color)
            if show_mean: plt.axvline(np.mean(all_stepsizes), color=MEANLINE_COLOR)
            plt.show()
            plt.title(title)
        else:
            ax.hist(all_stepsizes, bins=bins, color=color)
            if show_mean: ax.axvline(np.mean(all_stepsizes), color=MEANLINE_COLOR)
            ax.set_title(title)
        
        if show_mean:
            mean_line = mlines.Line2D([],[], color=MEANLINE_COLOR, linestyle="-")
            handles, labels = [mean_line], ['Mean']
            ax.legend(handles=handles, labels=labels)
    
        extrema[bpt] = {"min" : min(all_stepsizes), "max" : max(all_stepsizes)}
    return extrema

def visualize_stepsize_standard_deviation_per_mousegroup(stepsize_groups : list,
                                                         mousenames : list,
                                                         groupnames : list,
                                                         data_is_normal : bool,
                                                         significance : float,
                                                         colors : list,
                                                         bodyparts : list, 
                                                         savedir : str,
                                                         savename : str,
                                                         cutoff : float=STEPSIZE_MIN,
                                                         show_mean : bool=True,
                                                         show_mousename : bool=False,
                                                         save_figure : bool=True,
                                                         show_figure : bool=True,
                                                         save_mean_comparison_result : bool=True
        ):
    """
    Visualizes the standard deviation of step sizes for individual mice
    as individual points in a scatter plot, created side by side per mouse group
    per body part.

    :param list stepsize_groups: A list of list of dictionaries, holding info of step sizes.
    Each list is a group. Must be the exact same shape as mousenames, to hold matching names.
    :param list mousenames: A list of list of string, holding mouse names matching dictionaries
    passed in stepsize_groups. Each list is a group. Must be the same shape as stepsize_groups.
    :param list groupnames: A list of the names of group passed in as stepsize_groups and mousenames.
    len(groupnames) must match len(stepsize_groups) and len(mousenames).
    :param bool data_is_normal: Whether the Standard Deviations obtained from individual
    mice are distributed normally. Determines whether an unpaired t-test or Mann Whitney
    U-test is done on the difference in mean of SDs across groups.
    :param float significance: Significance level for the comparison of mean of SDs 
    between groups.
    :param list colors: A list of colors (string) for each mouse group. Must match len(groupnames).
    :param list bodyparts: Specifies which body part to create figures for.
    :param str savedir: Directory to save resulting figures.
    :param str savename: Name of file under which the resulting figure is saved.
    :param float cutoff: Cutoff for minimum step size to count, defaults to STEPSIZE_MIN
    :param bool show_mean: Whether to show mean SDs within each group, defaults to True
    :param bool show_mousename: Whether to annotate the mouse names to graphs, defaults to False
    :param bool save_figure: Whether to save a figure, defaults to True
    :param bool show_figure: Whether to show the figure, defaults to True
    :param bool save_mean_comparison_result: Whether to save the result of comparison
    for mean of SDs of mice per groups, under a file named 
    'meanComparisonOfSD_GROUPNAMES.txt" under savedir. Defaults to True.
    """
    if len(stepsize_groups) != len(groupnames):
        raise Exception(f"Length of stepsize_groups and groupnames must match but were: {len(stepsize_groups)} and {len(groupnames)}...")
    if len(stepsize_groups) != len(mousenames):
        raise Exception(f"Length of stepsize_groups and mousenames must match but were: {len(stepsize_groups)} and {len(mousenames)}...")
    if len(mousenames) != len(groupnames):
        raise Exception(f"Length of mousenames and groupnames must match but were: {len(mousenames)} and {len(groupnames)}...")
    if len(colors) != len(groupnames):
        raise Exception(f"Length of colors and groupnames must match but were: {len(colors)} and {len(groupnames)}...")
    

    all_results = ""
    stepsizes_dicts = []
    for mousename_grp, group in zip(mousenames, stepsize_groups):
        groupdict = {}
        # compute the std for each body part for each mouse
        for mousename, dictionary in zip(mousename_grp, group):
            stepsizes = analyze_mouse_gait.aggregate_stepsize_per_body_part(
                dictionaries=[dictionary], bodyparts=bodyparts, cutoff=cutoff)
            # compute the standard deviation and track it in a dict
            for bpt in bodyparts:
                bpt_dict = groupdict.get(bpt, {})
                if len(stepsizes[bpt]) == 0: 
                    bpt_dict[mousename] = np.nan
                else:
                    bpt_dict[mousename] = np.std(np.array(stepsizes[bpt]))
                groupdict[bpt] = bpt_dict
        stepsizes_dicts.append(groupdict)
    
    # create the figure
    for bpt in bodyparts:
        _, ax = plt.subplots(figsize=(3,6))
        bpt_stepsizes = [np.array(list(grpdicts[bpt].values())) 
                         for grpdicts in stepsizes_dicts]
        bpt_stepsizes = [stepsize[~np.isnan(stepsize)]
                         for stepsize in bpt_stepsizes]
        
        Y = np.concatenate(bpt_stepsizes)
        X = [[grpname]*len(stepsize) for (grpname, stepsize) in zip(groupnames, bpt_stepsizes)]
        X = np.concatenate(X)

        mousenames = np.array([mname for grpdicts in stepsizes_dicts 
                               for mname in grpdicts[bpt].keys() 
                               if not np.isnan(grpdicts[bpt][mname])])
        
        ylim_margin = np.ptp(Y)/6
        ax.scatter(x=X, y=Y)
        ax.set_ylim(top=   np.max(Y) + ylim_margin, 
                    bottom=np.min(Y) - ylim_margin)
        
        if show_mousename:
            for x_coord, y_coord, mname in zip(X, Y, mousenames):
                ax.annotate(mname, (x_coord, y_coord))

        for group_idx in range(len(groupnames)):
            
            if show_mean:
                mean_line = mlines.Line2D([],[], color='black', linestyle=MEAN_LINESTYLE)
                handles, labels = [mean_line], ['Mean']
                # set boundaries for mean bars
                barwidth = 1/3/(len(groupnames) + 1)
                xmax = (group_idx+1) / (len(groupnames) + 1) - barwidth
                xmin = (group_idx+1) / (len(groupnames) + 1) + barwidth
                
                ax.axhline(y=np.mean(bpt_stepsizes[group_idx]), 
                           xmax=xmax, xmin=xmin,
                           color=colors[group_idx], linestyle=MEAN_LINESTYLE)
                # also include the handles manually
                ax.legend(handles=handles, labels=labels)
            
            plt.xticks(range(-1, len(groupnames)+1), rotation=45)
        
        plt.suptitle(f"Standard Deviation Of \n{bpt} Step Size \n({', '.join(groupnames)})")
    
        if save_figure:
            plt.savefig(os.path.join(savedir, 
                                    savename.replace(".png", "") + bpt))
        if show_figure:
            plt.show()
        else:
            plt.close()
    
            # do a mean difference test
        filtered_stepsizes1, filtered_stepsizes2 = bpt_stepsizes[0], bpt_stepsizes[1]

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
        all_results += result_txt

    if save_mean_comparison_result:
        result_filename = f"meanComparisonOfSD_{'_'.join(groupnames)}.txt"
        with open(os.path.join(savedir, result_filename), 'w') as f:
            f.write(all_results)

def visualize_time_between_consecutive_landings(yamls : list, 
                                                savedir : str,
                                                save_prefix : str,
                                                binsize : int=10,
                                                save_figure : bool=True,
                                                show_figure : bool=True):
    # we could show, at once:
    # - the aggregate histograms for each pairs per mouse
    # - the aggregated histogram for each pair for all mice
    # - the change of landing time difference over time for each pair, 
    #   for each sequence

    def visualize_time_difference_per_mouse_per_pair(landings : dict,
                                                     mousename : str,
                                                     binsize : int=10):
        if len(landings) == 0: return
        
        first_landings = landings[list(landings.keys())[0]]
        _, axes = plt.subplots(ncols=1, nrows=len(first_landings)-1)
        # for each existing pairs of body parts
        bpt_pairts = [p for p in first_landings.keys() if p != "end"]
        for pair_idx, pair in enumerate(bpt_pairts):
            ax = axes if (len(bpt_pairts) == 1) else axes[pair_idx]

            all_landing_time_differences = []
            
            # aggregate all info on this pair from all sequences
            for sequence in landings.values():
                all_landing_time_differences.extend(sequence[pair])
            
            # plot it
            bins = range(0, 
                         (int(np.nanmax(all_landing_time_differences).item()) // binsize + 1) * binsize, 
                         binsize)
            ax.hist(all_landing_time_differences, bins=bins)
            ax.set_title(f"{pair.replace('_', ' ')}")
        
        plt.suptitle(f"Paw Landing Time Difference ({mousename}, binsize={binsize})")
        plt.tight_layout()
        
        if save_figure:
            mouse_subdir = os.path.join(savedir, mousename)
            if not os.path.exists(mouse_subdir):
                os.mkdir(mouse_subdir)
            plt.savefig(os.path.join(mouse_subdir, 
                                     f"{save_prefix}_TimeDifferencePerBodyPartPair"))

        if show_figure: plt.show()
        else: plt.close()
    
    def visualize_time_difference_per_pair_for_all(all_landings : list,
                                                   binsize : int=10):
        if len(all_landings) == 0: return
        
        first_file = all_landings[0]
        first_landings = first_file[list(first_file.keys())[0]]
        _, axes = plt.subplots(ncols=1, nrows=len(first_landings)-1)
        # for each existing pairs of body parts
        bpt_pairts = [p for p in first_landings.keys() if p != "end"]
        for pair_idx, pair in enumerate(bpt_pairts):
            ax = axes if (len(bpt_pairts) == 1) else axes[pair_idx]

            all_landing_time_differences = []
            
            # aggregate all info on this pair from all sequences
            for landings in all_landings:
                for sequence in landings.values():
                    all_landing_time_differences.extend(sequence[pair])
            
            # plot it
            bins = range(0, 
                         (int(np.nanmax(all_landing_time_differences).item()) // binsize + 1) * binsize, 
                         binsize)
            ax.hist(all_landing_time_differences, bins=bins)
            ax.set_title(f"{pair.replace('_', ' ')}")
        
        plt.suptitle(f"Paw Landing Time Difference Aggregating All Mice \n(binsize={binsize})")
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(os.path.join(savedir, 
                                     f"{save_prefix}_TimeDifferencePerBodyPartPairForAllMice"))

        if show_figure: plt.show()
        else: plt.close()
        
    for yaml in yamls:
        mousename = get_mousename(yaml)
        landings = read_consecutive_landing_time_yaml(yaml)

        # aggregate per mouse for each pairs
        visualize_time_difference_per_mouse_per_pair(landings=landings, 
                                                     mousename=mousename,
                                                     binsize=binsize)
    # aggregate altogether
    visualize_time_difference_per_pair_for_all(all_landings=[read_consecutive_landing_time_yaml(yaml)
                                                             for yaml in yamls], 
                                            binsize=binsize)
        
        



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
                             average_pawrest : bool=True, 
                             ignore_close_paw : float=None):
    """
    Takes in a data frame holding DLC data as well as an array of BSOID label
    for it, filtering out:
    - any non-locomotion label frame 
    - any locomotion label frame which is followed by movement above 'threshold'
    Returns the resulting data frame after filtering.
    If average_pawrest, we take each paw rest, averaging their location over the 
    the duration of its rest. Also return another data frame holding info in each row of: 
    [a body part of concern, start of its rest, end of its rest, 
     x coordinate averaged over the rest, y coordinate averaged over the rest].
    The corresponding column names are accessible as constants:
     [COL_BODYPART, COL_START, COL_LENGTH, COL_X_AVG, COL_Y_AVG].

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
    :param float ignore_close_paw: Whether to ignore consecutive paws closer than 'ignore_close_paw'
    post average. e.g. if a paw rest sequence is [5, 6, 15, 18] and 'ignore_close_paw' = 2,
    the '6' paw is ignored to yield [5, 15, 18] instead. Defaults to None, no ignoring.

    :returns pd.DataFrame filt_df: The data frame where corresponding paw rests are filtered out.
    :returns pd.DataFrame average_df: if 'average_pawrest' is true, the data frame specified in 
    the descrption of the function. Otherwise, None.
    """
    filt_df = df.copy(deep=True)
    # remove entries in df where movement of 'paws' is above threshold
    unique_bpts = np.unique(filt_df.columns.get_level_values('bodyparts'))
    for bpt in unique_bpts:
        if 'paw' in bpt:
            x, y = filt_df[bpt, 'x'].to_numpy(), filt_df[bpt, 'y'].to_numpy()
            movement = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            movement = np.insert(movement, 0, 0) # pad first position as no movement
            filt_df.loc[movement > threshold, [bpt]] = np.nan
        # completely remove entries for parts like snout and tailbase
        else:
            if not show_nonpaw:
                filt_df.loc[:, [bpt]] = np.nan
    # remove entries in filt_df that aren't locomotion bouts
    filt_df.loc[label != locomotion_labels[0], :] = np.nan
    
    # if average_pawrest: 1) identify paw rest sequences
    # 2) replace the entire sequence with its average
    if average_pawrest:
        # we also return an "averaged" data frame which holds, for every 
        # paw sequence: body part, first time stamp, length, x average, y average
        average_array = []
        for bpt in unique_bpts:
            latest_paw_pos = None
            if 'paw' in bpt:
                isnan_x = np.isnan(filt_df[bpt, 'x'].to_numpy())
                run_values, run_starts, run_lengths = find_runs(isnan_x)
                
                for nonnan_idx in np.where(run_values == False)[0]:
                    start = run_starts[nonnan_idx].item()
                    length = run_lengths[nonnan_idx].item()
                    end = start + length - 1
                    # get the average of x and y
                    avg_x = np.mean(filt_df.loc[start:end, (bpt, 'x')])
                    avg_y = np.mean(filt_df.loc[start:end, (bpt, 'y')])

                    # change the original data frame
                    filt_df.loc[start:end, (bpt, 'x')] = avg_x
                    filt_df.loc[start:end, (bpt, 'y')] = avg_y

                    # add a new row to the new dataframe
                    new_row = (bpt, start, length, avg_x, avg_y)
                        
                    if ignore_close_paw is not None:
                        if latest_paw_pos is None:
                            paw_too_close = False
                        else:
                            current_paw_pos = np.array([avg_x, avg_y])
                            paw_distance = np.sqrt(np.dot(current_paw_pos - latest_paw_pos, 
                                                          current_paw_pos - latest_paw_pos))
                            paw_too_close = (paw_distance < ignore_close_paw)
                        
                        if not paw_too_close: 
                            average_array.append(new_row)
                    else:
                        average_array.append(new_row)
                        
                    latest_paw_pos = np.array([avg_x, avg_y])
                        
        # make a dataframe out of it
        columns = [COL_BODYPART, COL_START, COL_LENGTH, COL_X_AVG, COL_Y_AVG]
        average_df = pd.DataFrame(data=average_array, 
                                  columns=columns)

    else:
        average_df = None
    return filt_df, average_df

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
    ['end', 'rightforepaw', 'leftforepaw', 'righthindpaw', 'lefthindpaw'].
    """
    with open(stepsize_yaml, 'r') as f:
        yaml_content = f.read()
    content = yaml.safe_load(yaml_content)
    return content

def read_consecutive_landing_time_yaml(file : str):
    """
    Read a yaml file holding "consecutive landing time" information
    for locomotion.

    :param str file: Path to a yaml file holding the info in question.
    :return Dict: A dictionary holding the data in question. Keys are the
    starting frame number, with entries: 
    ['end', body part pairs like 'lefthindpaw_righthindpaw'].
    """
    with open(file, 'r') as f:
        yaml_content = f.read()
    content = yaml.safe_load(yaml_content)
    return content