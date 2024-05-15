# Author: Akira Kudo (Copying A TON from the B-SOiD project)
# Created: 2024/03/19
# Last updated: 2024/05/15

# These code files were copied on 2024/03/19 from 
# Visualize_Feature_Distributions_Per_B_SOID_Bout_Cluster.ipynb and the code blocks
# are almost identical (except addition to be able to choose threshold & whether to
# do brute thresholding; and additional saving into csvs)!

import math
import os
from typing import List

import joblib
import numpy as np
import pandas as pd

from BSOID_code.adp_filt import adp_filt
from BSOID_code.bsoid_extract import bsoid_extract
from BSOID_code.bsoid_predict import bsoid_predict
from utils import Bodypart, generate_guessed_map_of_feature_to_data_index

FEATURE_SAVING_FOLDERS = "./B-SOID_features/results/features"
FEATURE_FILE_SUFFIX = "_features.npy"
LABEL_FILE_SUFFIX = "_labels.npy"

LABELED_FEATURE_FOLDERNAME = 'lbl_feats'
LABELED_FEATURE_CSV_SUFFIX = '_labeled_features.csv'

# Extract both labels and features from a csv file

def extract_label_and_feature_from_csv(filepath : str, pose : List[int],
                                       clf_path : str, fps : int=40,
                                       brute_thresholding=False, threshold=0.8,
                                       save_result=True, save_path=FEATURE_SAVING_FOLDERS,
                                       recompute=False,  load_path=FEATURE_SAVING_FOLDERS):
    """
    Given the path to a csv name to be analyzed, extracts and computes both the
    labels and features from it. Does check if previous computations have been
    made for both the labels and features, and if it finds both, will skip
    computation (unless instructed to recompute).
    :param str filepath: Path to csv file to be analyzed.
    :param List[int] pose: A list of indices relative to the csv that specifies
    which body part to use for computation. e.g. [0,1,2,6,7,8] will use those
    columns with those indices for computation.
    :param str clf_path: The path to the classifier-stored sav file, often
    called 'PREFIX_randomforest.sav'.
    :param int fps: Framerate for video. Defaults to 40, given Ellen's project.
    :param bool brute_thresholding: Whether to use brute (vs. adaptive) thresholding when 
    preprocessing DLC output based on likelihood pre-feature-extraction. Defaults to False.
    :param float threshold: Threshold to use when brute_thresholding is true. Defaults to 0.8.
    :param bool save_result: Whether to save the computed features under
    save_path. Defaults to true.
    :param str save_path: The folder to which we save computed features.
    Defaults to FEATURE_SAVING_FOLDERS.
    :param bool recompute: Whether to recompute features saved under load_path.
    :param str load_path: The path from which we attempt to load precomputed
    features. Defaults to FEATURE_SAVING_FOLDERS.
    """
    # first check relevant paths exist
    if not os.path.exists(filepath):
        raise Exception(f"Following file doesn't seem to exist: {filepath}...")
    if not os.path.exists(clf_path): 
        raise Exception(f"Couldn't find classifier: {filepath}...")
    if save_result and not os.path.exists(save_path): 
        raise Exception(f"Saving folder wasn't found: {save_path}...")
    if not recompute and not os.path.exists(load_path): 
        raise Exception(f"Folder holding data to potentially load wasn't found: {load_path}...")

    filename = os.path.basename(filepath).replace('.csv', '')
    clfname = os.path.basename(clf_path).replace('_randomforest.sav', '')
    feature_save_filename = clfname + "_" + filename + FEATURE_FILE_SUFFIX
    label_save_filename   = clfname + "_" + filename + LABEL_FILE_SUFFIX
    lbld_feats_filename   = clfname + "_" + filename + LABELED_FEATURE_CSV_SUFFIX

    if not recompute:
        # attempt fetch
        feature = fetch_precomputed_from_npy(os.path.join(load_path, feature_save_filename))
        label   = fetch_precomputed_from_npy(os.path.join(load_path,   label_save_filename))
        # if it doesn't work, recompute
        
        if feature is None or label is None: recompute = True

    if recompute:
        # read data from csv and filter data
        file_j_df = pd.read_csv(filepath, low_memory=False)
        file_j_processed, _ = adp_filt(file_j_df, pose, brute_thresholding, threshold)
        # code assumes multiple data processed at once, though I will only process one
        final_labels = []
        labels_fs = []
        # extraction & prediction, while also computing the merged feature at once
        feature, feats_new = compute_merged_features_from_csv_data(file_j_processed, fps)
        clf = load_classifier(clf_path)
        labels = bsoid_predict(feats_new, clf)
        # padding of labels
        # invert the content of labels
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
        # create all -1 padding with second size being length of longest label list
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        # populate a frame-shifted stack of predictions that total to the whole video    
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
        # Frameshift arrangement of predicted labels
        for l_fs in labels_fs:
            labels_fs2 = []
            for l in range(math.floor(fps / 10)):
                labels_fs2.append(l_fs[l])
            final_labels.append(np.array(labels_fs2).flatten('F'))
        label = final_labels[0]

    if save_result:
        print(f"Saving results to: {save_path}")
        try:
            # first create the folder to hold labeled features
            lbld_feats_save_folder = os.path.join(save_path, LABELED_FEATURE_FOLDERNAME)
            if not os.path.exists(lbld_feats_save_folder):
                os.mkdir(lbld_feats_save_folder)
                
            # save the features
            feature_save_path = os.path.join(save_path, feature_save_filename)
            if os.path.exists(feature_save_path):
                print(f"Overwriting feature: - {feature_save_filename}")
            np.save(feature_save_path, feature)
            
            # then the label
            label_save_path = os.path.join(save_path, label_save_filename)
            if os.path.exists(label_save_path):
                print(f"Overwriting label: - {label_save_filename}")
            np.save(label_save_path, label)

            # then save the labeled feature
            lbld_feats_save_path = os.path.join(lbld_feats_save_folder, lbld_feats_filename)
            if os.path.exists(lbld_feats_save_path):
                print(f"Overwriting labeled feature: - {lbld_feats_filename}")
            # specify which bodypart we consider - predicted from pose
            bodyparts = [Bodypart(pose_val // 3) 
                         for pose_val in pose 
                         if Bodypart(pose_val // 3) not in bodyparts]
            
            feature_to_index_map = generate_guessed_map_of_feature_to_data_index(
                bodyparts, short=True
            )
            # then create column names from those info
            column_header = ["label"] + ["_tofill_"] * len(feature_to_index_map)
            for feature_name, feature_idx in feature_to_index_map.items():
                column_header[feature_idx + 1] = feature_name
            # finally populate a dataframe with the given data
            label = label[:feature.shape[1]] # truncate labels by the number of features
            # len(labels) > len(features) as artifact of padding - have to double check
            df_data = np.concatenate((np.expand_dims(label, axis=0), feature), 
                                    axis=0).T
            saved_df = pd.DataFrame(data=df_data, columns=column_header)
            saved_df.to_csv(lbld_feats_save_path)

        except Exception as e:
            print(e)
            print("Moving on...")

    return label, feature


# helper to load a classifier from a sav file
def load_classifier(path):
    with open(path, 'rb') as fr:
        [_, _, _, clf, _, _] = joblib.load(fr)
    return clf

# load previous file that holds computed values in the result folder
# path : A path to the file that holds pre-computed values.
# Returns: The values, or None if the file of interest did not exist.
def fetch_precomputed_from_npy(path):
    feature = None
    if os.path.exists(path):
        print(f"Fetching precomputed {path}!")
        feature = np.load(path, allow_pickle=True)
        print(f"Success!")
    return feature


def compute_merged_features_from_csv_data(data : np.ndarray, fps : int):
    """
    Takes a single data np.ndarray and extracts & computes a merged feature set
    from it.
    :param np.ndarray data: The data numpy array from which we extract the
    features.
    :param int fps: Fps for the video we are dealing with.
    """
    feature = None
    # compute features from data - features is n lists of 10fps binned features
    # e.g. with a 40 fps video, we get 4 lists of 10 fps binned feature calculated
    #      something like [0,4], [1,5], [2,6] and [3,7] created from [0,1,2,3,4,5,6,7]
    #      where each number is the starting frame of a 10 fps bin
    premerged_feature = bsoid_extract([data], fps)
    # so we want to interleave them to get the final feature time series
    feature = np.empty((premerged_feature[0].shape[0], # each row is a feature
                        sum(feat.shape[1] for feat in premerged_feature)),  # get total length of all 10 fps data
                        dtype=premerged_feature[0].dtype)
    stride = len(premerged_feature) # the same feature vector is entered with a stride of how many features there are
    # e.g. merging [0,3], [1,4] and [2,5] into [0,1,2,3,4,5] is done with a stride of 3
    for i, feat in enumerate(premerged_feature):
        feature[:, i::stride] = feat
    return feature, premerged_feature