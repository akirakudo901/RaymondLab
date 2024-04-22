# Author: Akira Kudo
# Created: 2024/04/04
# Last Updated: 2024/04/22

import os

import numpy as np

from feature_analysis_and_visualization.visualization.quantify_labels import quantify_label_occurrence_and_length_distribution
from bsoid_io.utils import read_BSOID_labeled_csv

YAC_FOLDER = r"C:\Users\mashi\Desktop\temp\YAC"
WT_FOLDER = r"C:\Users\mashi\Desktop\temp\WT"
Q175_FOLDER = r"C:\Users\mashi\Desktop\temp\Q175\BSOID csvs"

Q175_LABEL_FOLDER = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\feats_labels"

def convert_folder_to_list_of_label(folder : str):
    csvs = [os.path.join(folder, file) for file in os.listdir(folder) 
            if '.csv' in file]
    labels = [read_BSOID_labeled_csv(csv)[0] for csv in csvs]
    return labels

def convert_folder_of_labels_to_list_of_label(folder : str, iter : int=None):
    npys = [os.path.join(folder, file) for file in os.listdir(folder) 
            if '.npy' in file and 'labels' in file and (iter is None or iter in file)]
    labels = [np.load(npy) for npy in npys]
    return labels

yac_labels  = convert_folder_to_list_of_label(YAC_FOLDER)
wt_labels  = convert_folder_to_list_of_label(WT_FOLDER)
q175_labels  = convert_folder_to_list_of_label(Q175_FOLDER)

q175_labels = convert_folder_of_labels_to_list_of_label(Q175_LABEL_FOLDER, iter="2060000")

quantify_label_occurrence_and_length_distribution(
    group_of_labels=[
        # yac_labels, wt_labels, 
        q175_labels
        ],
    group_names=[
        # "YAC128", "WT", 
        "Q175"
        ], 
    use_logscale=False,
    save_dir=r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID\COMPUTED\results\figures\Akira_Apr082024",
    save_name="Q175.png",
    save_figure=True
)

