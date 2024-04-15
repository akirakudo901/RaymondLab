# Author: Akira Kudo
# Created: 2024/03/31
# Last updated: 2024/04/02

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from label_behavior_bits.preprocessing import repeating_numbers

LOCOMOTION_LABEL = 38

def analyze_mouse_gate(df : pd.DataFrame, 
                       label : np.ndarray, 
                       bodyparts : list):
    
    # obtain all locomotion-labeled sequences from labels
    n_list, idx, lengths = repeating_numbers(labels=label)
    locomotion_within_array = (np.array(n_list) == LOCOMOTION_LABEL)
    locomotion_idx = np.array(idx)[locomotion_within_array]
    locomotion_lengths = np.array(lengths)[locomotion_within_array]
    
    for bpt_idx, bpt in enumerate(bodyparts):
        x, y = df[bpt, 'x'].to_numpy(), df[bpt, 'y'].to_numpy()
        diff = np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y)))
        
        # TODO REMOVE
        # RENDER THE GATE INTO A TIMESERIES
        
        # TODO REMOVE END