# Author: Akira Kudo
# Created: 2024/03/31
# Last updated: 2024/05/22

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...label_behavior_bits.preprocessing import repeating_numbers
from ..visualization.visualize_mouse_gait import find_locomotion_sequences

"""
Take a single csv. From it, we can:
1) Extract all locomotion bouts
2) Visualize info:
 - How many locomotion bouts there are
 - How many of each length of locomotion bouts there are
 - What's the distribution of the distance between two locomotion bouts (curious if bouts are broken down but close together)
 - What's the distribution of locomotion for every 5 min interval (just curious)
3) Extract the paw stopping based on speed
4) For every paw stopping identified as single bout, find the average paw position within that stop
5) Graph such paw positions over time in:
 - a static plot
 - video, to confirm of its correctness
6) Save this data into a csv, and:
 - calculate the distance between each paw movement
 - calculate the distance between each body part
 and so on.
7) Visualize:
- distance traveled in a single bout
- max / min / average speed of those single bouts
- frequency of bouts with different speeds
8) Maybe do analysis based on those variables
"""

LOCOMOTION_LABELS = [38]

# def extract_mouse_gait(df : pd.DataFrame, 
#                        label : np.ndarray, 
#                        bodyparts : list, 
#                        locomotion_label=LOCOMOTION_LABELS):
    

def analyze_mouse_gait(df : pd.DataFrame, 
                       label : np.ndarray, 
                       bodyparts : list, 
                       locomotion_label=LOCOMOTION_LABELS):
    
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
