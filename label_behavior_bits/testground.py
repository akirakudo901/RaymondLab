
# import os

# import cv2
# import pandas as pd

# from create_labeled_behavior_bits import label_bodyparts_on_single_frame


# FPS = 40
# MIN_DESIRED_BOUT_LENGTH = 200
# COUNTS = 5
# OUTPUT_FPS = 30
# TRAILPOINTS = 0

# PCUTOFF = 0
# DOTSIZE = 7
# COLORMAP = "rainbow" # obtained from config.yaml on DLC side
# # we exclude "belly" as it isn't used to classify in this B-SOID
# BODYPARTS = ["snout",        "rightforepaw", "leftforepaw", 
#                 "righthindpaw", "lefthindpaw",  "tailbase", "belly"] 

# FILE_OF_INTEREST = r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
# LABELED_PREFIX = r"Feb-27-2024labels_pose_40Hz"

# OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\B-SOID STUFF\BoutVideoBits\labeled"

# FRAME_DIR     = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\pngs", FILE_OF_INTEREST.replace(".csv", ""))
# OUTPUT_PATH   = os.path.join(OUTPUT_FOLDER, FILE_OF_INTEREST.replace(".csv", ""))
# DATA_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv",      FILE_OF_INTEREST)
# LABELED_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\BSOID\Feb-27-2024", 
#                                 LABELED_PREFIX + FILE_OF_INTEREST)

# def extract_label_from_labeled_csv(labeled_csv_path):
#     df = pd.read_csv(labeled_csv_path, low_memory=False)
#     labels = df.loc[:,'B-SOiD labels'].iloc[2:].to_numpy()
#     return labels


# labels = extract_label_from_labeled_csv(LABELED_CSV_PATH)

# Dataframe = pd.read_csv(DATA_CSV_PATH, index_col=0, header=[0,1,2], skip_blank_lines=False, low_memory=False)

# l = 0

# image = os.listdir(FRAME_DIR)[0]

# read_img = cv2.imread(os.path.join(FRAME_DIR, image))
# cv2.imshow('Before', read_img) #TODO REMOVE
# cv2.waitKey(0)
# labeled_img = label_bodyparts_on_single_frame(read_img,
#                                             index=l,
#                                             Dataframe=Dataframe,
#                                             pcutoff=PCUTOFF,
#                                             dotsize=DOTSIZE,
#                                             colormap=COLORMAP,
#                                             bodyparts2plot=BODYPARTS,
#                                             trailpoints=TRAILPOINTS)
# cv2.imshow('After', labeled_img) #TODO REMOVE
# cv2.waitKey(0)

import os

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import extract_label_from_labeled_csv, replace_one_length_frame_with_matching_neighbors

FILE_OF_INTEREST = r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
LABELED_PREFIX = r"Feb-27-2024labels_pose_40Hz"

LABELED_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\BSOID\Feb-27-2024", 
                                LABELED_PREFIX + FILE_OF_INTEREST)

START, DURATION = 0, 30
END = START + DURATION

labels = extract_label_from_labeled_csv(LABELED_CSV_PATH)
filtered_labels = replace_one_length_frame_with_matching_neighbors(labels)

print(np.where(filtered_labels==9)[0][:10])

# fig, ax = plt.subplots()
# ax.step(range(START, END),          labels[START:END], label= "Pre-filtering", color='g')
# ax.step(range(START, END), filtered_labels[START:END], label="Post-filtering", color='b')
# ax.legend()
# plt.grid(visible=True)
# plt.show()