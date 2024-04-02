# Author: Akira Kudo
# Created: 2024/03/21
# Last Updated: 2024/03/26

import os

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from ..utils import find_runs

def quantify_label_occurrence_and_length_distribution(labels : np.ndarray):
    """
    Quantifies and visualizes the occurrence of labels in terms of:
    - the number of occurrence of the specific label
    - the number of occurrence of consecutive groups of label
    - the distribution of those runs

    :param np.ndarray labels: The labels we want to quantify & visualize.
    """
    R = np.linspace(0, 1, len(np.unique(labels)))
    color_per_label = plt.cm.get_cmap("Spectral")(R)

    def visualize_frequency_from_numpy_array(labels : np.ndarray,
                                             plot_title : str,
                                             use_logscale : bool=True):
        # get the number of occurrence per label
        unique_labels = np.unique(labels).tolist()
        unique_label_counts = [np.sum(labels == unique_l) for unique_l in unique_labels]
        # then sort them out according to their frequency
        sorted_pairs = sorted(zip(unique_label_counts, unique_labels, color_per_label))
        sorted_unique_labels = [ul for _, ul, _ in sorted_pairs]
        sorted_unique_label_counts = [ulc for ulc, _, _ in sorted_pairs]
        sorted_color_per_label = [cpl for _, _, cpl in sorted_pairs]
        # finally plot a bar graph for it
        _, ax = plt.subplots(figsize=(20,6))
        ax.bar(range(len(sorted_unique_label_counts)), 
               sorted_unique_label_counts, 
               width=1, 
               color=sorted_color_per_label)
        ax.set_xticks(range(len(sorted_unique_labels)), labels=sorted_unique_labels)
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
        one_percent_height = len(labels) / 100
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

        plt.title(plot_title)
        if use_logscale: plt.yscale('log')
        plt.show()
    
    # first visualize the number of occurrence per label into an overlaid plot
    visualize_frequency_from_numpy_array(labels=labels, 
                                         plot_title="Raw Occurrence Of Labels",
                                         use_logscale=False)
    # then visual;ize the runs and their distributions
    run_values, run_starts, run_lengths = find_runs(labels)
    # first show the bout lengths
    # plot_bout_length(labels, use_logscale=True)
    # then show their frequency of occurrence
    visualize_frequency_from_numpy_array(labels=run_values, 
                                         plot_title="Occurrence Of Behavior Snippets", 
                                         use_logscale=False)



def plot_bout_length(labels : np.ndarray,
                     use_logscale : bool=True):
    FIGSIZE_X, FIGSIZE_Y = 6, max(6, len(np.unique(labels))//2)
    YTICK_FONTSIZE = 7.5

    run_values, _, run_lengths = find_runs(labels)
    unique_labels = np.unique(run_values)

    R = np.linspace(0, 1, len(unique_labels))
    color = plt.cm.get_cmap("Spectral")(R)

    fig = plt.figure(facecolor='w', edgecolor='k', figsize=(FIGSIZE_X, FIGSIZE_Y))
    upperbound, lowerbound = max(run_lengths), min(run_lengths)

    for i, l in enumerate(unique_labels):
        plt.subplot(len(unique_labels), 1, i + 1)

        l_lengths = run_lengths[np.where(l == run_values)]

        num_bins = min(50, (upperbound - lowerbound)+1)
        plt.hist(l_lengths,
                bins=np.linspace(lowerbound, upperbound, num=num_bins),
                range=(lowerbound, upperbound),
                color=color[i])

        fig.suptitle("Length of features")
        plt.xlim(lowerbound, upperbound)
        plt.yticks(fontsize=YTICK_FONTSIZE)
        # if specified, use log scale for ticks
        if use_logscale and l_lengths.size != 0:
            plt.yscale('log')
        # add a label of which label group to each histogram
        plt.gca().set_title(f'{l}', loc='right', y=-0.2)

        if i < len(unique_labels) - 1:
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.show()


    

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

    l_ltN, l_freq, l_perc, total_perc = quantify_labels_happening_less_than_N(labels, n=N)
    print(f"Labels that happen less than {N} times were:")
    for unique_label, label_frequency, label_percentage in zip(l_ltN, l_freq, l_perc):
        print(f"- {unique_label:<2} : {label_frequency:>2} times, or {round(label_percentage, 3):>4}%")
    print(f"All of them total to {round(total_perc, 3)}% of the data.")