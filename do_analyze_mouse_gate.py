# Author: Akira Kudo
# Created: 2024/04/01
# Last updated: 2024/05/14

import os
import sys

# I will learn about proper packaging and arrangement later...
sys.path.append(r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID")
sys.path.append(r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut")

from BSOID.feature_analysis_and_visualization.visualization.visualize_mouse_gate import visualize_mouse_gait_speed, visualize_mouse_paw_rests_in_locomomotion
from BSOID.bsoid_io.utils import read_BSOID_labeled_csv
from BSOID.label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

from DeepLabCut.temper_with_csv_and_hdf.data_filtering.identify_paw_noise import identify_bodypart_noise_by_impossible_speed
from DeepLabCut.temper_with_csv_and_hdf.data_filtering.filter_based_on_boolean_array import filter_based_on_boolean_array

FILE_OF_INTEREST = r"312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
LABELED_PREFIX = r"Mar-10-2023labels_pose_40Hz"

MOUSETYPE_FOLDER = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Leland B-SOID YAC128 Analysis" + \
                   r"\YAC128\YAC128" # r"\Q175\WT"

LABELED_CSV_PATH = os.path.join(MOUSETYPE_FOLDER,  
                                "CSV files", "BSOID", #r"csv/BSOID/Feb-27-2024"
                                LABELED_PREFIX + FILE_OF_INTEREST)

# LABELED_CSV_PATH = r"C:\Users\mashi\Desktop\temp\Q175\BSOID csvs\Apr-08-2024labels_pose_40Hz20230107131118_363453_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"
LABELED_CSV_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\Feb232023\WT\CSV files\BSOID\Mar-10-2023labels_pose_40Hz351201_m3DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"

BODYPARTS = [
    'righthindpaw', 'lefthindpaw', 
    'rightforepaw', 'leftforepaw', 
    'snout', 'tailbase'
    ]

# read csv
label, df = read_BSOID_labeled_csv(LABELED_CSV_PATH)
filt_label = filter_bouts_smaller_than_N_frames(label=label, n=5)

noise_df = identify_bodypart_noise_by_impossible_speed(bpt_data=df,
                                                       bodyparts=BODYPARTS,
                                                       start=0, 
                                                       end=df.shape[0], # number of rows
                                                       savedir=None, 
                                                       save_figure=False, 
                                                       show_figure=False)

# filter based on obtained noise info
filtered_df = df 
for bpt in BODYPARTS:
    bpt_bool_arr = noise_df[(bpt, "loc_wrng")].to_numpy()

    filtered_df = filter_based_on_boolean_array(
        bool_arr=bpt_bool_arr, df=filtered_df, bodyparts=[bpt], filter_mode="linear"
    )

visualize_mouse_paw_rests_in_locomomotion(df=filtered_df,
                                          label=filt_label,
                                          bodyparts=BODYPARTS,
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