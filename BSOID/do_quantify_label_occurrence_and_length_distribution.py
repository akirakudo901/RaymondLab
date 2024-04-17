# Author: Akira Kudo
# Created: 2024/04/04
# Last Updated: 2024/04/17

import os

from feature_analysis_and_visualization.visualization.quantify_labels import quantify_label_occurrence_and_length_distribution
from bsoid_io.utils import read_BSOID_labeled_csv

YAC_FOLDER = r"C:\Users\mashi\Desktop\temp\YAC"
WT_FOLDER = r"C:\Users\mashi\Desktop\temp\WT"
Q175_FOLDER = r"C:\Users\mashi\Desktop\temp\Q175\BSOID csvs"

def convert_folder_to_list_of_label(folder : str):
    csvs = [os.path.join(folder, file) for file in os.listdir(folder) 
            if '.csv' in file]
    labels = [read_BSOID_labeled_csv(csv)[0] for csv in csvs]
    return labels

yac_labels  = convert_folder_to_list_of_label(YAC_FOLDER)
wt_labels  = convert_folder_to_list_of_label(WT_FOLDER)
q175_labels  = convert_folder_to_list_of_label(Q175_FOLDER)

quantify_label_occurrence_and_length_distribution(
    group_of_labels=[
        # yac_labels, wt_labels, 
        q175_labels
        ],
    group_names=[
        # "YAC128", "WT", 
        "Q175"
        ], 
    use_logscale=False
)

