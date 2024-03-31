# Author: Akira Kudo
# Created: 2024/03/28
# Last updated: 2024/03/31

import numpy as np

import matplotlib.pyplot as plt

from feature_analysis_and_visualization.behavior_groups import BehaviorGrouping

def plot_behavior_group_transition(label : np.ndarray, 
                                   network_name : str, 
                                   start : int, 
                                   end : int):
    # translate label into corresponding behavioral groups
    bg = BehaviorGrouping(network_name=network_name)
    lbl_2_bhvrl_int = np.vectorize(bg.label_to_behavioral_group_int)
    behavior_grouping_label = lbl_2_bhvrl_int(label)
    
    # then plot the result while giving legends
    unique_behavior_groups_str = bg.get_behavior_groups()
    unique_behavior_groups_int = [bg.grouping_str_to_grouping_int[bg_str] 
                                  for bg_str in unique_behavior_groups_str]
    # decide colors
    R = np.linspace(0, 1, len(unique_behavior_groups_int))
    color = plt.cm.get_cmap("Spectral")(R)
    
    start, end = max(start, 0), min(end, len(label))
    _, ax = plt.subplots(figsize=(10, 5))
    for idx, bg_int in enumerate(unique_behavior_groups_int):
        x = np.where(behavior_grouping_label == bg_int)[0]
        shown_x = x[np.logical_and(x <= end, x >= start)]
        if len(shown_x) != 0:
            ax.bar(shown_x, [1]*len(shown_x), width=1, color=color[idx], 
                label=bg.grouping_int_to_grouping_str[bg_int])
    ax.legend()
    plt.show()