# Author: Akira Kudo
# Code for preprocessing DLC and B-SOID labels before rendering 
# them onto videos.

import numpy as np
import pandas as pd
from tqdm import tqdm

############
# DLC data #
############

def adp_filt_bsoid_style(datax : np.ndarray, 
                         datay : np.ndarray, 
                         data_lh : np.ndarray, 
                         brute_thresholding : bool=False):
    """
    Preprocesses data extracted from a DLC csv to replace low-likelihoods
    DLC position predictions with the most recent high-likelihood prediction.
    Turns data to what is used by B-SOID for making feature computation and 
    predictions.
    :param np.ndarray datax: x position data, [timepoints x number of bodyparts]
    :param np.ndarray datay: y position data, [timepoints x number of bodyparts]
    :param np.ndarray data_lh: likelihood data, [timepoints x number of bodyparts]
    :param bool brute_thresholding: Whether to use brute thresholding or the adaptive
    thresholding that B-SOID originally used.
    """
    datax_filt, datay_filt = np.zeros_like(datax), np.zeros_like(datay)
    
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float32))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        ##################
        # ADDED BY AKIRA
        if brute_thresholding:
          llh = 0.8
        ##################
        data_lh_float = data_lh[:, x].astype(np.float32)
        datax_filt[0, x], datay_filt[0, x] = datax[0, x], datay[0, x]
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                datax_filt[i, x], datay_filt[i, x] = datax_filt[i - 1, x], datay_filt[i - 1, x]
            else:
                datax_filt[i, x], datay_filt[i, x] = datax[i, x], datay[i, x]
    datax_filt = np.array(datax_filt).astype(np.float32)
    datay_filt = np.array(datay_filt).astype(np.float32)
    return datax_filt, datay_filt




#################
# B-SOID labels #
#################

def extract_label_from_labeled_csv(labeled_csv_path : str):
    """
    Extracts labels from an already labeled csv.
    :returns np.ndarray labels: Labels extracted, dim=(timepoints, )
    """
    df = pd.read_csv(labeled_csv_path, low_memory=False)
    # first two rows of label are just padded labels
    labels = df.loc[:,'B-SOiD labels'].iloc[2:].to_numpy()
    return labels

def replace_one_length_frame_with_matching_neighbors(label : np.ndarray):
    """
    Replaces one-length bouts with the labels of its neighbors, if
    both of its neighbor (left and right) agree in label.
    The process is ran starting from both ends of the array, and 
    only those that agree from both sides are actually filtered.
    :param np.ndarray label: Label in question, dim=(timepoints, ).
    :returns np.ndarray filtered: Filtered label with same size as 'label'.
    """
    forward, backward = label.copy(), label.copy()
    for i in range(1, len(forward) - 1):
        if forward[i-1] == forward[i+1]: forward[i] = forward[i-1]
    # reverse check
    for i in range(len(backward) - 2, 0, -1):
        if backward[i-1] == backward[i+1]: backward[i] = backward[i-1]
    # replace 'forward' content with label content where forward[x] != backward[x]
    unmatch = forward != backward
    forward[unmatch] = label[unmatch]
    return forward

def filter_bouts_smaller_than_N_frames(label : np.ndarray, 
                                       n : int):
    """
    For all i between 1 and n inclusive, replaces i length bouts 
    with the labels of its neighbors, if both of its i neighbors 
    (left and right) all agree in label.
    The process is ran starting from 1, and from both ends of 
    the array, where for each run, only those labels that agree 
    from both sides are actually filtered.
    :param np.ndarray label: Label to filter.
    :param int n: Largest bout length to consider for replacement.
    """
    
    for boutlength in range(1, n+1):
        forward, backward = label.copy(), label.copy()
        # forward check; window for boutlength = 2 will be like:
        # -|-||x|x|0|o|x|x||-|- where: windowcenter = 0,
        # p1 = first two xs, p3 = last two xs and
        # all of p2 (0 and o) are replaced if all xs match.
        for windowcenter in range(boutlength, len(forward) - boutlength):
            p1_start, p1_end = windowcenter - boutlength, windowcenter
            p2_start, p2_end = windowcenter             , windowcenter + boutlength
            p3_start, p3_end = windowcenter + boutlength, windowcenter + 2 * boutlength
            # if all surrounding entries have the same entry
            if np.all(forward[p1_start] == forward[p1_start : p1_end]) and \
               np.all(forward[p1_start] == forward[p3_start : p3_end]):
                forward[p2_start : p2_end] = forward[p1_start : p1_end]
        # backward check
        for windowcenter in range(len(forward) - boutlength - 1, boutlength - 1, -1):
            p1_start, p1_end = windowcenter + 1                 , windowcenter + boutlength + 1
            p2_start, p2_end = windowcenter - boutlength + 1    , windowcenter + 1
            p3_start, p3_end = windowcenter - 2 * boutlength + 1, windowcenter - boutlength + 1
            # if all surrounding entries have the same entry
            if np.all(backward[p1_start] == backward[p1_start : p1_end]) and \
               np.all(backward[p1_start] == backward[p3_start : p3_end]):
                backward[p2_start : p2_end] = backward[p1_start : p1_end]
        # replace 'forward' content with label content where forward[x] != backward[x]
        unmatch = forward != backward
        forward[unmatch] = label[unmatch]
        # go to the next iteration, treating current 'forward' as label
        label = forward
    return label