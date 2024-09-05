# Author: Akira Kudo
# Created: 2024/03/21
# Last Updated: 2024/08/29

import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy
import seaborn as sns

from ..analysis.analyze_time_spent_per_label import compute_time_spent_per_label_per_group_from_numpy_array, FRAMECOUNT, GROUPNAME, MOUSENAME, PERCENTAGE
from ..utils import find_runs

LABEL_DELIMITER_VALUE = -1

def quantify_label_occurrence_and_length_distribution(
        group_of_labels : List[List[np.ndarray]],
        group_names : List[str],
        save_dir : str,
        save_name : str,
        label_to_name : dict=None,
        use_logscale : bool=False,
        save_figure : bool=True,
        show_figure : bool=True
        ):
    """
    For each group of labels, quantifies and visualizes the occurrence 
    of labels in terms of:
    - the number of occurrence of the specific label
    - the number of occurrence of consecutive groups of label
    - the distribution of those runs

    :param List[List[np.ndarray]] group_of_labels: A list of groups of labels,
    which are each themselves a list of labels, that we want to quantify & visualize.
    :param List[str] group_names: The name of groups we wanna observe.
    :param str save_dir: Path to the directory to save figures.
    :param str save_name: Name of file to given when saving it, png. Will append the
    description of each generated figure when saving.
    :param dict label_to_name: A dict mapping labels to their name to be displayed on
    the figure.
    :param bool use_logscale: Whether to use log scale for y-axis of visualization.
    :param bool save_figure: Whether to save figures. Defaults to True.
    :param bool show_figure: Whether to show figures. Defaults to True.
    """
    if not save_name.endswith('.png'): save_name += '.png'

    all_labels = np.concatenate([np.concatenate(gol) for gol in group_of_labels])
    unique_labels = np.unique(all_labels)
    R = np.linspace(0, 1, len(unique_labels))
    color_per_label = plt.cm.get_cmap("Spectral")(R)

    def visualize_frequency_from_numpy_array(group_of_labels : List[List[np.ndarray]],
                                             plot_title : str,
                                             save_name : str,
                                             use_logscale : bool=True):
        # first group every label group into a single array
        to_be_grouped_labels = [[np.append(lbl, LABEL_DELIMITER_VALUE) for lbl in gol] 
                                for gol in group_of_labels]
        grouped_labels = [np.concatenate(gol) for gol in to_be_grouped_labels]
        
        # then create the plots
        _, axes = plt.subplots(len(group_of_labels), 1, figsize=(20,3*len(group_of_labels)))
        for i, (grouped_lbl, name) in enumerate(zip(grouped_labels, group_names)):
            ax = axes if (len(group_of_labels) == 1) else axes[i]
            # get the number of occurrence per label
            unique_label_counts = [np.sum(grouped_lbl == unique_l) for unique_l in unique_labels]
            # then sort them out according to their frequency
            sorted_pairs = sorted(zip(unique_label_counts, unique_labels, color_per_label))
            sorted_unique_labels = [ul for _, ul, _ in sorted_pairs]
            sorted_unique_label_counts = [ulc for ulc, _, _ in sorted_pairs]
            sorted_color_per_label = [cpl for _, _, cpl in sorted_pairs]
            # if given, rename the labels using label_to_name
            if label_to_name is not None:
                sorted_unique_labels = [label_to_name[ul.item()] 
                                        for ul in sorted_unique_labels]
            # finally plot a bar graph for it
            ax.bar(range(len(sorted_unique_label_counts)), 
                sorted_unique_label_counts, 
                width=1, 
                color=sorted_color_per_label)
            ax.set_xticks(range(len(sorted_unique_labels)))
            ax.set_xticklabels(labels=sorted_unique_labels)
            # draw lines indicating at a set interval the height of different bars
            
            def draw_line_and_mark_y_val_at(axis, height_label_pair : list):
                # obtain a transform that locates the label at the correct position
                trans = transforms.blended_transform_factory(
                    axis.get_yticklabels()[0].get_transform(), axis.transData)
                for h, l in height_label_pair:
                    axis.axhline(h, color='green', linewidth=0.5)
                    axis.text(0, h, f"{l}", color="green", transform=trans, 
                            ha="right", va="center")

            # draw some additional lines with respect to total label sizes 
            one_percent_height = len(grouped_lbl) / 100
            additional_line_props = []
            additional_line_props.append( (one_percent_height, "1%"))
            interval_percentage = 5
            for percentage in range(interval_percentage, 100, interval_percentage):
                if (percentage * one_percent_height > 
                max(unique_label_counts) + interval_percentage * one_percent_height):
                    break
                else:
                    additional_line_props.append( (percentage*one_percent_height, f"{percentage}%") )
            draw_line_and_mark_y_val_at(axis=ax, height_label_pair=additional_line_props)
            ax.yaxis.tick_right()
            ax.set_ylabel(name, labelpad=30)
            if use_logscale: ax.yscale('log')
        plt.suptitle(plot_title)
        if save_figure:
            plt.savefig(os.path.join(save_dir, save_name))
        if show_figure:
            plt.show()
        else:
            plt.close()
    
    # first visualize the number of occurrence per label into an overlaid plot
    visualize_frequency_from_numpy_array(group_of_labels=group_of_labels, 
                                         plot_title="Raw Occurrence Of Labels",
                                         save_name=save_name.replace('.png', '_Raw_Occurrence.png'),
                                         use_logscale=use_logscale)
    # then visualize the runs and their distributions
    group_run_values = [[find_runs(lbl)[0] for lbl in gol]
                        for gol in group_of_labels]
    # then show their frequency of occurrence
    visualize_frequency_from_numpy_array(group_of_labels=group_run_values, 
                                         plot_title="Occurrence Of Behavior Snippets", 
                                         save_name=save_name.replace('.png', '_Behavior_Snippets.png'),
                                         use_logscale=use_logscale)

def quantify_labels_happening_less_than_N(labels : np.ndarray, n : int):
    """
    Identify unique labels that happen less than N times in 'labels', 
    counting their proportion in relation to the whole label set.
    :param np.ndarray labels: numpy array of label we examine, 
    shape of [num_timepoints,].
    :param int n: Examine whether labels happen less often than n.
    :returns :
    """
    labels_less_than_n, label_frequencies, label_percentages = [],[],[]
    total_percentage = 0
    
    for unique_label in np.unique(labels):
        occurrence_count = np.sum(labels == unique_label)
        occurrence_percentage = occurrence_count / len(labels) * 100
        if occurrence_count < n:
            labels_less_than_n.append(unique_label)
            label_frequencies.append(occurrence_count)
            label_percentages.append(occurrence_percentage)
            total_percentage += occurrence_percentage
    
    return labels_less_than_n, label_frequencies, label_percentages, total_percentage

def visualize_label_occurrences_heatmaps(
        group_of_labels : List[List[np.ndarray]],
        group_names : List[str],
        mousenames : List[List[str]],
        labels_to_check : List[str],
        ylabel : str,
        display_threshold : float,
        save_dir : str,
        save_name : str,
        vmin : float=None,
        vmax : float=None,
        xlabel : str="Label Groups",
        title : str=None,
        save_figure : bool=True,
        show_figure : bool=True, 
        figsize : tuple=(12,6)
        ):
    """
    Takes in a list of 'groups of labels' with corresponding names, 
    generating a heatmap for the ocurrence of labels. One heatmap is
    created per group, which rows correspond to mice and columns to labels.

    :param List[List[np.ndarray]] group_of_labels: A list of groups of labels,
    which are each themselves a list of labels, that we want to quantify & visualize.
    :param List[str] group_names: The name of groups we wanna observe.
    :param List[List[str]] mousenames: List of groups of names of mice, each
    corresponding to the numpy array contained in nparray_groups. Expected to be
    the same shape as nparray_groups.
    :param List[int] labels_to_check: List of integer indicating which label groups
    to consider when storing their time spent, defaults to all labels.
    :param str ylabel: Label for y-axis.
    :param float display_threshold: A threshold in percentage. Any label with 
    occurrence less than this value in any of the mice is not displayed to 
    improve visibility.
    :param str save_dir: Directory for saving figure.
    :param str save_name: Name of saved figure.
    :param float vmin: Minimum value to which we anchor the heatmap, defaults 
    to minimum of all features rendered in a single heatmap.
    :param float vmax: Maximum value to which we anchor the heatmap, defaults 
    to maximum of all features rendered in a single heatmap.
    :param str xlabel: Label for x-axis, defaults to "Label Groups".
    :param str title: Figure title, defaults to '{xlabel} per {ylabel}'.
    :param bool save_figure: Whether to save figure, defaults to True.
    :param bool show_figure: Whether to show figure, defaults to True.
    :param bool figsize: Size of the rendered figure, defaults to [12, 6] in inches
    which I believe is fairly big to capture all the groups.
    """
    # make sure the number of groups and their names match
    if len(group_of_labels) != len(group_names):
        raise Exception("group_of_labels and group_names have to be the same length...")
    
    # set default title if not given
    if title is None:
        title = f'{xlabel} per {ylabel}'
    
    # create a data frame holding information on label occurrence
    df = compute_time_spent_per_label_per_group_from_numpy_array(
            nparray_groups=group_of_labels,
            mousenames=mousenames,
            group_names=group_names,
            save_path=None,
            label_groups=labels_to_check,
            save_csv=False, show_message=True
            )
    # for each group, one heatmap
    _, axes = plt.subplots(len(group_of_labels), 1, figsize=figsize)
    for group_idx, groupname in enumerate(group_names):
        ax = axes if (len(group_names) == 1) else axes[group_idx]
        # we obtain information to plot
        group_individuals = df[df[GROUPNAME] == groupname]
        individual_mousenames = group_individuals[MOUSENAME].str.replace(
            '_', '').drop_duplicates()
        group_columns = df.columns[np.logical_not(np.isin(df.columns, [MOUSENAME, GROUPNAME]))]
        # we exclude any label occurring less than display_threshold to increase visibility
        all_entry_less_than_thresh = np.any(df.loc[PERCENTAGE, group_columns] > display_threshold, axis=0)
        group_columns = group_columns[all_entry_less_than_thresh]
        label_occurrences = group_individuals.loc[PERCENTAGE, group_columns]
        # create a heatmap where each row is an individual and
        # each column is a label group
        if vmin is None: vmin = np.min(label_occurrences.to_numpy())
        if vmax is None: vmax = np.max(label_occurrences.to_numpy())
        sns.heatmap(label_occurrences.to_numpy(), annot=True, fmt='.0f', 
                    cmap='coolwarm', linewidths=0, ax=ax, cbar=False,
                    xticklabels=group_columns.str.replace('group', ''), 
                    yticklabels=individual_mousenames,
                    vmin=vmin, vmax=vmax)
        # half since we have twice as many rows (framecount & percentage) as mice
        ax.set_title(f'{groupname} (n={len(group_individuals)//2})') 
        # set labels selectively
        ax.set_xlabel(xlabel)
        if group_idx == 0: ax.set_ylabel('Individuals')

    # set the other settingss
    plt.suptitle(title)
    plt.tight_layout()

    if save_figure:
        plt.savefig(os.path.join(save_dir, save_name))
    
    if show_figure:
        plt.show()
    else:
        plt.close()

def quantify_max_percentage_per_mouse_per_labels(labels : list,
                                                 mousenames : list,
                                                 label_groups : list,
                                                 savedir : str,
                                                 savename : str,
                                                 threshold : float=3.0,
                                                 show_message : bool=True,
                                                 save_result : bool=True):
    max_percentages = {}

    # create a data frame holding information on label occurrence
    all_labels_combined = np.concatenate(labels)
    df = compute_time_spent_per_label_per_group_from_numpy_array(
        nparray_groups=[[lbl] for lbl in labels] + [[all_labels_combined]],
        mousenames=[[msnm] for msnm in mousenames] + [["All"]],
        group_names=mousenames + ["All"],
        save_path=None,
        label_groups=label_groups,
        save_csv=False, show_message=True
        )
    # compute the maximum percentage of label coverage in all mice for each label
    perc_df = df.loc[PERCENTAGE]
    for col in perc_df.columns:
        if col != MOUSENAME and col != GROUPNAME:
            max_perc = np.max(perc_df.loc[PERCENTAGE, col])
            max_percentages[col] = max_perc
    
    result_message = "Maximum percentage in mice & percentage across all mice:\n"
    for grpname, perc in max_percentages.items():
        perc_across_all_mice = perc_df[perc_df[MOUSENAME] == "All"].loc[PERCENTAGE, grpname]
        result_message += f"-{grpname}: {perc}; {perc_across_all_mice}\n"
    
    result_message += f"\n Entries with less then {threshold}% max in any mouse are:"
    total_perc_below_thresh = 0
    for grpname, perc in max_percentages.items():
        if perc < threshold:
            result_message += f"{grpname.replace('group', '')}({round(perc, 2)}); "
            perc_across_all_mice = perc_df[perc_df[MOUSENAME] == "All"].loc[PERCENTAGE, grpname]
            total_perc_below_thresh += perc_across_all_mice
    
    result_message += f"\nThese groups total {round(total_perc_below_thresh, 2)}% of all labels!"
    
    if show_message:
        print(result_message)
        
    if save_result:
        savepath = os.path.join(savedir, savename)
        with open(savepath, 'w') as f:
            f.write(result_message)

if __name__ == "__main__":
    LABEL_NUMPY_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID_related\feature_extraction\results"
    FILE_OF_INTEREST = "312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000"
    LABEL_NUMPY_FILENAME = f"Feb-23-2023_{FILE_OF_INTEREST}_labels.npy"

    LABEL_NUMPY_PATH = os.path.join(LABEL_NUMPY_DIR, LABEL_NUMPY_FILENAME)
    
    labels = np.load(LABEL_NUMPY_PATH, allow_pickle=True)

    N = len(labels) * 0.01 // 1

    # l_ltN, l_freq, l_perc, total_perc = quantify_labels_happening_less_than_N(labels, n=N)
    # print(f"Labels that happen less than {N} times were:")
    # for unique_label, label_frequency, label_percentage in zip(l_ltN, l_freq, l_perc):
    #     print(f"- {unique_label:<2} : {label_frequency:>2} times, or {round(label_percentage, 3):>4}%")
    # print(f"All of them total to {round(total_perc, 3)}% of the data.")

    