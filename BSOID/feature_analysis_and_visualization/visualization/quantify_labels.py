# Author: Akira Kudo
# Created: 2024/03/21
# Last Updated: 2024/04/23

import os
from typing import List

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from ..utils import find_runs

LABEL_DELIMITER_VALUE = -1

def quantify_label_occurrence_and_length_distribution(
        group_of_labels : List[List[np.ndarray]],
        group_names : List[str],
        save_dir : str,
        save_name : str,
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

    