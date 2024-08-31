# Author: Akira Kudo
# Created: 2024/08/30
# Last Updated: 2024/08/30

import numpy as np
import os

from BSOID.bsoid_io.utils import read_BSOID_labeled_features
from BSOID.feature_analysis_and_visualization.behavior_groups import BehaviorGrouping
from BSOID.label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

def prep_label(csv_path : str, start : int=0, end : int=30*40*60):
    """
    Preps csv by reading and extracting label, then truncating the 
    label to from start to end.
    """
    label, _ = read_BSOID_labeled_features(csv_path)
    return label[start:end+1]

def convert_label_to_int_metalabel(label : np.ndarray, 
                                behavior_grouping : BehaviorGrouping):
    return np.fromiter([behavior_grouping.label_to_behavioral_group_int(val) 
                        for val in label],
                        dtype=np.int32)

if __name__ == "__main__":
    YAC_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\SortedForMarja\BSOiD_Feature\YAC128"
    CSV_SAVE_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\SortedForMarja\MetaLabel"

    CSV_FOLDER_LIST = [os.path.join(YAC_FOLDER, "HD_filt"),
                       os.path.join(YAC_FOLDER, "WT_filt")]

    CSV_PATH_LIST = [
        [os.path.join(folder, file) for file in os.listdir(folder)
        if (file.endswith(".csv"))]
        for folder in CSV_FOLDER_LIST 
        ]

    LABEL_LIST = [
        [prep_label(csv) for csv in group]
        for group in CSV_PATH_LIST
    ]

    NETWORK_NAME = 'Feb-23-2023'
    YAML_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\SortedForMarja\MetaLabel\behavior_groups_Akira.yml"
    FILTER_MIN_SIZE = 5

    FILT_LABEL_LIST = [
        [filter_bouts_smaller_than_N_frames(label, n=FILTER_MIN_SIZE) for label in group]
        for group in LABEL_LIST
    ]

    bg = BehaviorGrouping(network_name=NETWORK_NAME, yaml_path=YAML_PATH)
    groupings = bg.load_behavior_groupings(network_name=NETWORK_NAME, yaml_path=YAML_PATH)

    # convert the labels into integer meta labels
    METALABEL_LIST = [
        [convert_label_to_int_metalabel(lbl, bg) for lbl in group]
        for group in FILT_LABEL_LIST
        ]
    
    for metalabel_group, filename_group in zip(METALABEL_LIST, CSV_PATH_LIST):
        for metalabel, filename in zip(metalabel_group, filename_group):
            print(f"Processing file: {os.path.basename(filename)}!")
            orig_filename = os.path.basename(filename)
            save_filename = "META_" + orig_filename.replace('_labeled_features.csv', '.txt')

            if "HD_filt" in filename:
                full_savepath = os.path.join(CSV_SAVE_FOLDER, "HD_filt", save_filename)
            elif "WT_filt" in filename:
                full_savepath = os.path.join(CSV_SAVE_FOLDER, "WT_filt", save_filename)
            np.savetxt(full_savepath, metalabel, fmt=r'%.0d')