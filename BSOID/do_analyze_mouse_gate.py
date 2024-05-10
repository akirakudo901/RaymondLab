# Author: Akira Kudo
# Created: 2024/04/01
# Last updated: 2024/05/09

import os

from feature_analysis_and_visualization.visualization.visualize_mouse_gate import visualize_mouse_gait_speed, visualize_mouse_paw_rests_in_locomomotion
from bsoid_io.utils import read_BSOID_labeled_csv
from label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

FILE_OF_INTEREST = r"312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
LABELED_PREFIX = r"Mar-10-2023labels_pose_40Hz"

MOUSETYPE_FOLDER = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Leland B-SOID YAC128 Analysis" + \
                   r"\YAC128\YAC128" # r"\Q175\WT"

LABELED_CSV_PATH = os.path.join(MOUSETYPE_FOLDER,  
                                "CSV files", "BSOID", #r"csv/BSOID/Feb-27-2024"
                                LABELED_PREFIX + FILE_OF_INTEREST)

# LABELED_CSV_PATH = r"C:\Users\mashi\Desktop\temp\Q175\BSOID csvs\Apr-08-2024labels_pose_40Hz20230107131118_363453_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"
LABELED_CSV_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\Feb232023\WT\CSV files\BSOID\Mar-10-2023labels_pose_40Hz351201_m3DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"

# read csv
label, df = read_BSOID_labeled_csv(LABELED_CSV_PATH)
filt_label = filter_bouts_smaller_than_N_frames(label=label, n=5)

visualize_mouse_paw_rests_in_locomomotion(df=df,
                                          label=filt_label,
                                          bodyparts=['righthindpaw', 'lefthindpaw', 
                                                     'rightforepaw', 'leftforepaw', 
                                                     'snout', 'tailbase'
                                                     ],
                                          length_limits=(80, None),
                                          plot_N_runs=5,
                                          locomotion_label=[38]
                                          )

# visualize_mouse_gait_speed(df=df, 
#                      label=filt_label, 
#                      bodyparts=['righthindpaw', 'lefthindpaw', 
#                                 # 'rightforepaw', 'leftforepaw', 
#                                 # 'snout', 'tailbase'
#                                 ],
#                      length_limits=(40, None),
#                     #  plot_N_runs=5,
#                      locomotion_label=[29,30])