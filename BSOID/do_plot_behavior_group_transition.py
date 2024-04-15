# Author: Akira Kudo
# Created: 2024/04/04
# Last Updated: 2024/04/04

# import os

# import numpy as np

# from BSOID_related.feature_visualization.quantify_labels import quantify_label_occurrence_and_length_distribution

# LABEL_NUMPY_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID_related\feature_extraction\results"
# FILE_OF_INTEREST = "312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000"
# LABEL_NUMPY_FILENAME = f"Feb-23-2023_{FILE_OF_INTEREST}_labels.npy"

# LABEL_NUMPY_PATH = os.path.join(LABEL_NUMPY_DIR, LABEL_NUMPY_FILENAME)

# labels = np.load(LABEL_NUMPY_PATH, allow_pickle=True)

# quantify_label_occurrence_and_length_distribution(labels)

import pandas as pd

from feature_analysis_and_visualization.visualization.plot_behavior_group_transition import plot_behavior_group_transition
from label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

from feature_analysis_and_visualization.analysis.analyze_time_spent_per_label import analyze_time_spent_per_label_between_groups

NETWORK_NAME = 'Feb-23-2023'
CSV_PATH = r"C:\Users\mashi\Desktop\temp\YAC\Mar-10-2023labels_pose_40Hz312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"

SAVEDIR = r"C:\Users\mashi\Desktop\temp\figures"

START, END = 0, 500
FILTER_SIZE = 5

df = pd.read_csv(CSV_PATH, index_col=[0], 
                 header=[0,1,2])
label = df.iloc[:, 0].to_numpy().T
plot_behavior_group_transition(label=label, 
                               network_name=NETWORK_NAME,
                               start=START,
                               end=END, 
                               savename=f'pre-filtering_{START}~{END}.png',
                               savedir=SAVEDIR, 
                               savefigure=True)

filtered_label = filter_bouts_smaller_than_N_frames(label, n=FILTER_SIZE)
plot_behavior_group_transition(label=filtered_label, 
                               network_name=NETWORK_NAME,
                               start=START,
                               end=END, 
                               savename=f'post-{FILTER_SIZE}filtering_{START}~{END}.png',
                               savedir=SAVEDIR, 
                               savefigure=True)

# all_labels, group1, group2 = analyze_time_spent_per_label_between_groups(
#     [CSV_PATH, CSV_PATH], [CSV_PATH], groups_to_check=[1,2,3])

# framecount1, percentage1 = group1[0], group1[1]
# framecount2, percentage2 = group2[0], group2[1]