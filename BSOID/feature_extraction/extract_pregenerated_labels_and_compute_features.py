# Author: Akira Kudo
# Created: 2024/04/17
# Last Updated: 2024/04/17

import os

import joblib
import numpy as np

from .extract_label_and_feature_from_csv import compute_merged_features_from_csv_data, fetch_precomputed_from_npy, FEATURE_SAVING_FOLDERS, FEATURE_FILE_SUFFIX, LABEL_FILE_SUFFIX

def extract_pregenerated_labels_and_compute_features(predictionsPath, filename,
                              clf_sav_path : str, fps=40,
                              save_result=True, save_path=FEATURE_SAVING_FOLDERS,
                              recompute=False,  load_path=FEATURE_SAVING_FOLDERS):
    """
    Given a specific csv name that was analyzed, extracts a numpy.ndarray of
    frame-shifted labels stored within a 'predictions.sav' file at time of
    previous analysis. Simultaneously, extracts frameshifted data and uses it to
    compute frameshifted corresponding features.
    :param str filename: Name of csv that was analyzed in order to extract the
    labels of interest.
    :param str clf_sav_path: The path to the classifier-stored sav file, often
    called 'PREFIX_randomforest.sav'. Used to name storing files consistently.
    :param int fps: Framerate for video. Defaults to 40, given Ellen's project.
    :param bool save_result: Whether to save the computed features under
    save_path. Defaults to true.
    "param str save_path: The folder to which we save computed features.
    Defaults to FEATURE_SAVING_FOLDERS.
    :param bool recompute: Whether to recompute features saved under load_path.
    :param str load_path: The path from which we attempt to load precomputed
    features. Defaults to FEATURE_SAVING_FOLDERS.
    """

    # this is meant to be used as reference name
    # as of right now, it should hopefully match that of extract_label_and_feature_from_csv
    clfname = os.path.basename(clf_sav_path).replace('_randomforest.sav', '')
    feature_save_filename = clfname + "_" + filename + FEATURE_FILE_SUFFIX
    label_save_filename   = clfname + "_" + filename + LABEL_FILE_SUFFIX

    label = None; feature = None
    # As found under bsoid_app.predict:
    # listOfFolderPassed, listOfFolderForEachFileName, fileNames, positionData, labels = joblib.load(predictionsPath)
    _, _, fileNames, data, labels = joblib.load(predictionsPath)
    fileNames = [os.path.basename(fn) for fn in fileNames]
    # search filename as contained within fileNames
    idx_of_interest = None
    for idx, flnm in enumerate(fileNames):
      if filename in flnm: idx_of_interest = idx; break

    # if we can't find a matching result, return default values (Nones)
    if idx_of_interest is None:
        print("No matching filename found in extract_pregenerated_labels_and_compute_features.")
        return label, feature

    # otherwise, get label first
    label = labels[idx_of_interest]
    # if recompute is set to false, look for fetching
    if not recompute:
        feature = fetch_precomputed_from_npy(os.path.join(load_path, feature_save_filename))

        if feature is None:
             print(f"Failed fetching a precompute set of feature from {load_path}; computing.")
             recompute = True

    if recompute:
        # then also get frameshifted data
        data_of_interest = data[idx_of_interest]
        # compute the feature from it
        feature, _ = compute_merged_features_from_csv_data(data_of_interest, fps=fps)

        if save_result:
            print(f"Saving results to: {save_path}")
            np.save(os.path.join(save_path, feature_save_filename), feature)
            np.save(os.path.join(save_path,   label_save_filename),   label)

    return label, feature