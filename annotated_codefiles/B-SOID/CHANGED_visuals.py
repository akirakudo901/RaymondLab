"""
Visualization functions and saving plots.
"""

from enum import Enum
import itertools
import os
import time
from typing import List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import numpy as np
import pandas as pd
import seaborn as sn

from bsoid_umap.config import *

matplotlib_axes_logger.setLevel('ERROR')


Bodypart = Enum('Bodypart', ["SNOUT", "RIGHTFOREPAW", "LEFTFOREPAW", 
                             "RIGHTHINDPAW", "LEFTHINDPAW", 
                             "TAILBASE", "BELLY"])
# seems like the old b-soid scans six body parts (likely excluding belly, as the bsoid paper does? Though not sure)
bodyparts = [Bodypart.SNOUT, Bodypart.RIGHTFOREPAW, Bodypart.LEFTFOREPAW, 
             Bodypart.RIGHTHINDPAW, Bodypart.LEFTHINDPAW, Bodypart.TAILBASE] #Bodypart.BELLY
bodypart_pairs = list(itertools.combinations(bodyparts, 2))

# a list of features to visualize
# [15 x distance between body pairs] and
# [15 x    angle between body pairs] and
# [6  x distance change per body part]
# HAVE TO ENSURE OF THE CORRECT POSITION

def relative_placement_name(bp1 : Bodypart, bp2 : Bodypart):
    return "Relative {} to {} placement".format(bp1.name, bp2.name) 

def relative_angle_name(bp1 : Bodypart, bp2 : Bodypart):
    return "Relative angle from {} to {}".format(bp1.name, bp2.name)

def displacement_name(bp : Bodypart):
    return "Displacement of {}".format(bp.name)

EDUCATED_GUESS = dict()
i = 0
for bp1, bp2 in bodypart_pairs:
    EDUCATED_GUESS[relative_placement_name(bp1, bp2)] = i
    i += 1
for bp1, bp2 in bodypart_pairs:
    EDUCATED_GUESS[relative_angle_name(bp1, bp2)] = i
    i += 1
for bp in bodyparts:
    EDUCATED_GUESS[displacement_name(bp)] = i
    i += 1


def plot_feats(feats: list, labels: list):
    """
    :param feats: list, features for multiple sessions
     (Akira) Each element in the list seems to be of the form:
      [number of features x timestamps(i.e. bins)]
    :param labels: list, labels for multiple sessions
     (Akira) Each element in the list seems to be of the form:
    [timestamps(i.e. bins)]
    """
    result = isinstance(labels, list)
    timestr = time.strftime("_%Y%m%d_%H%M")

    # a list of features to visualize
    # [15 x distance between body pairs] and
    # [15 x    angle between body pairs] and
    # [6  x distance change per body part]
    # HAVE TO ENSURE OF THE CORRECT POSITION
    relative_placement_pairs = [
        (Bodypart.SNOUT, Bodypart.RIGHTFOREPAW),
        (Bodypart.SNOUT, Bodypart.RIGHTHINDPAW),
        (Bodypart.RIGHTFOREPAW, Bodypart.LEFTFOREPAW),
        (Bodypart.SNOUT, Bodypart.TAILBASE)
    ]
    relative_angle_pairs = [
        (Bodypart.SNOUT, Bodypart.TAILBASE)
    ]
    displacement_bodyparts = [ Bodypart.SNOUT, Bodypart.TAILBASE ]
    
    feat_ls = [relative_placement_name(*pair) for pair in relative_placement_pairs] + \
              [    relative_angle_name(*pair) for pair in     relative_angle_pairs] + \
              [      displacement_name( bdpt) for bdpt in   displacement_bodyparts]
    
    if result:
        # feats is a list of features for each session
        num_sessions = len(feats)
        for session_idx in range(0, num_sessions):
            
            session_feats = np.array( feats[session_idx])
            session_label = np.array(labels[session_idx])
            
            helper_plot_feats(feature=session_feats, label=session_label, 
                              figurename='sess{}_'.format(session_idx+1),
                              list_of_feature=feat_ls)
    else:
        helper_plot_feats(feature=feats, label=labels, figurename="sess0_",
                          list_of_feature=feat_ls)


def helper_plot_feats(feature : np.ndarray, label : np.ndarray, figurename : str,
                      list_of_feature : List[str]):
    # fetch the colors
    R = np.linspace(0, 1, len(np.unique(label)))
    color = plt.cm.get_cmap("Spectral")(R)

    # for each feature in this session
    for feature_name in list_of_feature:
        curr_feature_index = EDUCATED_GUESS[feature_name]
        print(feature[curr_feature_index, :])
        # calculate feature mean and sd
        ft_mean = np.mean(feature[curr_feature_index, :]); ft_std = np.std(feature[curr_feature_index, :])
        
        fig = plt.figure(facecolor='w', edgecolor='k')
        # for up to the number of unique labels - 1
        for curr_unique_label_idx in range(0, len(np.unique(label))-1):
            # plot one of the subplots in the i+1th row
            plt.subplot(len(np.unique(label)), 1, curr_unique_label_idx + 1)
            # for some, we plot from 0 to three std above mean
            if np.min(feature[curr_feature_index]) >= 0:
                lowerbound = 0; upperbound = ft_mean + 3 * ft_std
            # for others, we plot three std to both sides of the mean
            else:
                lowerbound = ft_mean - 3 * ft_std
                upperbound = ft_mean + 3 * ft_std
            # shared plotting code
            plt.hist(feature[curr_feature_index, label == curr_unique_label_idx],
                        bins=np.linspace(lowerbound, upperbound, num=50),
                        range=(lowerbound, upperbound),
                        color=color[curr_unique_label_idx], density=True)
            fig.suptitle("{} pixels".format(feature_name))
            plt.xlim(lowerbound, upperbound)
            if curr_unique_label_idx < len(np.unique(label)) - 1:
                plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        figurename = figurename + 'feat{}_hist'.format(curr_feature_index + 1)
        fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (figurename, timestr, '.svg'))))

    plt.show()

def main():
    return


if __name__ == '__main__':
    main()
