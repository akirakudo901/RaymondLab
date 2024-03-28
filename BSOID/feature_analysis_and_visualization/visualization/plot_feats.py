# Author: Akira Kudo
# Created: -
# Last updated: 2024/03/28
"""
Visualization functions and saving plots.
"""

from datetime import datetime
import os
import pytz
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from feature_extraction.utils import relative_placement_name, relative_angle_name, displacement_name

def plot_feats(feats: list, labels: list,
               show_specific_groups : List[int],
               relative_placement_pairs : list,
               relative_angle_pairs : list,
               displacement_bodyparts : list,
               feature_to_index_map : dict,
               figure_save_dir : str,
               csv_name : str,
               show_figure : bool=True,
               save_figure : bool=True,
               use_logscale : bool=False,
               brute_thresholding : bool=False):
    """
    :param feats: list, features for multiple sessions. Each element in the
    list seems to be of the form: [number of features x timestamps(i.e. bins)]
    :param labels: list, labels for multiple sessions. Each element in the
    list seems to be of the form: [timestamps(i.e. bins)]
    :param List[int] show_specific_groups: The list of group labels to
    specifically show for this figure. Some might be skipped due to not having
    any occurrence in this data. If an empty list is passed, render everything.
    :param List[List[Bodypart]] relative_placement_pairs: A list of pairs of
    Bodypart we want to plot the relative placement features for.
    :param List[List[Bodypart]] relative_angle_pairs: A list of pairs of
    Bodypart we want to plot the relative angle features for.
    :param List[Bodypart] displacement_bodyparts: A list of Bodypart which
    feature of displacement between consecutive frames is of our interest.
    :param Dict[str, int] feature_to_index_map: A mapping of names for feature
    name to their guessed index position within the raw feature vector.
    Checking the code cell "Defining body parts for rendering features" might
    be somewhat more informative?
    :param str csv_name: The name of the csv used to generate these plots.
    :param str figure_save_dir: Directory to which we save figures.
    :param bool use_logscale: Whether to use logarithmic scale on the y-axis.
    Defaults to false.
    :param bool brute_thresholding: Whether the features passed to this function
    were calculated from data filtered using brute thresholding with fixed threshold 
    value, or B-SOID's builtin adaptive thresholding. Defaults to False.
    """
    feat_ls = [relative_placement_name(*pair) for pair in relative_placement_pairs] + \
              [    relative_angle_name(*pair) for pair in     relative_angle_pairs] + \
              [      displacement_name( bdpt) for bdpt in   displacement_bodyparts]

    result = isinstance(labels, list)

    newfilename = get_newname(show_specific_groups, labels)
    newfilename = get_mousename(csv_name) + newfilename
    figure_subdir = os.path.join(figure_save_dir, newfilename)
    if save_figure and not os.path.exists(figure_subdir):
        os.mkdir(figure_subdir)

    if result:
        # feats is a list of features for each session
        num_sessions = len(feats)
        for session_idx in range(0, num_sessions):

            session_feats = np.array( feats[session_idx])
            session_label = np.array(labels[session_idx])

            helper_plot_feats(feature=session_feats, label=session_label,
                              show_specific_groups=show_specific_groups,
                              figurename='',
                              list_of_feature=feat_ls,
                              feature_to_index_map=feature_to_index_map,
                              figure_save_dir=figure_subdir,
                              csv_name=csv_name,
                              show_figure=show_figure,
                              save_figure=save_figure,
                              use_logscale=use_logscale,
                              brute_thresholding=brute_thresholding)
    else:
        helper_plot_feats(feature=feats, label=labels,
                          show_specific_groups=show_specific_groups,
                          figurename="",
                          list_of_feature=feat_ls,
                          feature_to_index_map=feature_to_index_map,
                          figure_save_dir=figure_subdir,
                          csv_name=csv_name,
                          show_figure=show_figure,
                          save_figure=save_figure,
                          use_logscale=use_logscale,
                          brute_thresholding=brute_thresholding)

def get_mousename(filename : str):
    # Extracting information from the filename using regular expressions
    searched = re.search(r'(\d+)(_?)[mf](\d+)', filename)
    if searched:
       return searched[0].replace('_', '')
    else:
       return "No_Match"

def get_newname(show_specific_groups, labels):
  vancouver_time = datetime.now(pytz.timezone('America/Vancouver'))
  Ybd_HM = vancouver_time.strftime("%Y%b%d_%H%M")
  if len(show_specific_groups) == 0:
    str_shown_groups = "all"
  else:
    unique_labels = np.unique(labels).tolist()
    # set up what groups to show in sorted order
    ssg_copy = [i for i in show_specific_groups if i in unique_labels]; ssg_copy.sort()
    # aggregate groups of labels we wanna show together
    min_ssg, max_ssg = min(ssg_copy), max(ssg_copy)
    shown_groups, in_a_group = [], False
    for i in range(min_ssg, max_ssg+1):
      if i in ssg_copy and not in_a_group:
        start, end, in_a_group = i, i, True
      elif i in ssg_copy:
        end = i
      elif i not in ssg_copy and in_a_group:
        shown_groups.append((start, end)); in_a_group = False
    if in_a_group:
      shown_groups.append((start, end))
    # we consider two namings: show all ranges, or show what is excluded from ranges
    show_in_ranges = '_'.join([str(start)+"to"+str(end) for start, end in shown_groups])
    show_by_exclusion = str(min_ssg) + "to" + str(max_ssg) + "_not_" + \
                        '_'.join([str(i) for i in range(min_ssg, max_ssg+1) if i not in ssg_copy])
    # choose the shorter name
    str_shown_groups = show_in_ranges if len(show_in_ranges) < len(show_by_exclusion) else show_by_exclusion
  return f"{Ybd_HM}_{str_shown_groups}"

def helper_plot_feats(feature : np.ndarray, label : np.ndarray,
                      show_specific_groups : List[int],
                      figurename : str,
                      list_of_feature : List[str],
                      feature_to_index_map : Dict[str, int],
                      figure_save_dir : str,
                      csv_name : str,
                      show_figure : bool=True,
                      save_figure : bool=True,
                      use_logscale : bool=False, 
                      brute_thresholding : bool=False):

    vancouver_time = datetime.now(pytz.timezone('America/Vancouver'))
    timestr = vancouver_time.strftime("_%Y%m%d_%H%M")

    # if we pass an empty show_specific_groups, use all available labels
    if len(show_specific_groups) == 0:
        unique_labels_to_render = np.unique(label).tolist()
    else:
        unique_labels_to_render = [l for l in np.unique(label).tolist() if l in show_specific_groups]

    unique_labels_to_render.sort()
    print(f"Rendering the following labels: {unique_labels_to_render}!")
    num_unique_labels_to_render = len(unique_labels_to_render)

    # properties like font size
    FIGSIZE_X, FIGSIZE_Y = 6, max(6, num_unique_labels_to_render//3)
    YTICK_FONTSIZE = 7.5

    #######TEMPORAL##########
    feature_time_length = feature.shape[1]
    label = label[:feature_time_length]
    #######TEMPORAL##########

    # fetch the colors
    R = np.linspace(0, 1, num_unique_labels_to_render)
    color = plt.cm.get_cmap("Spectral")(R)

    # for each feature in this session
    for feature_name in list_of_feature:
        curr_feature_index = feature_to_index_map[feature_name]
        curr_feature_array = feature[curr_feature_index, :]
        # calculate feature mean and sd
        ft_mean, ft_std = np.mean(curr_feature_array), np.std(curr_feature_array)
        ft_min,  ft_max =  np.min(curr_feature_array), np.max(curr_feature_array)

        fig = plt.figure(facecolor='w', edgecolor='k', figsize=(FIGSIZE_X, FIGSIZE_Y))

        # for up to the number of unique labels - 1
        for curr_unique_label_idx, unique_label in enumerate(unique_labels_to_render):
            feature_with_current_label = feature[curr_feature_index,
                                                 label == unique_label]

            # plot one of the subplots in the i+1th row
            plt.subplot(num_unique_labels_to_render, 1, curr_unique_label_idx + 1)

            # set lower & uppperbound around std and mean
            lowerbound, upperbound = ft_min - ft_std, ft_max + ft_std

            # shared plotting code
            plt.hist(feature_with_current_label,
                     bins=np.linspace(lowerbound, upperbound, num=50),
                     range=(lowerbound, upperbound),
                     color=color[curr_unique_label_idx], density=True)
            fig.suptitle("{} pixels".format(feature_name))
            plt.xlim(lowerbound, upperbound)
            plt.yticks(fontsize=YTICK_FONTSIZE)
            # if the feature array isn't empty, put in log scale as needed
            if use_logscale and feature_with_current_label.size != 0:
              plt.yscale('log')
            # add title to each plot as name of label group & number of
            # timepoints with that label
            plt.gca().set_title(
                f'{unique_label} [{feature_with_current_label.size}]',
                                loc='right', y=-0.2)

            if curr_unique_label_idx < num_unique_labels_to_render - 1:
                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        full_figurename = figurename + '{}_hist'.format(feature_name)
        if save_figure:
          fig.savefig(os.path.join(figure_save_dir,
                                  str.join('', (full_figurename,
                                                "_Logscale_" if use_logscale else "",
                                                timestr))))
          # also save a text file with its name being the name of the mouse csv
          # we used to generate these plots
          txtname = csv_name.replace(".csv", ".txt")
          with open(os.path.join(figure_save_dir, txtname), 'w') as f:
            f.write(csv_name)
            f.write(f"Use bruteforce vs. adaptive threshold: {'bruteforce' if brute_thresholding else 'adaptive'}")
            f.write(f"Groups to be shown: {show_specific_groups}")

    if show_figure: plt.show()