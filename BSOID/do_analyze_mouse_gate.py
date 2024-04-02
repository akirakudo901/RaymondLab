# Author: Akira Kudo
# Created: 2024/04/01
# Last updated: 2024/04/01

import os

import pandas as pd

from feature_analysis_and_visualization.visualization.visualize_mouse_gate import visualize_mouse_gate

FILE_OF_INTEREST = r"312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
LABELED_PREFIX = r"Mar-10-2023labels_pose_40Hz"

MOUSETYPE_FOLDER = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Leland B-SOID YAC128 Analysis" + \
                   r"\YAC128\YAC128" # r"\Q175\WT"

LABELED_CSV_PATH = os.path.join(MOUSETYPE_FOLDER,  
                                "CSV files", "BSOID", #r"csv/BSOID/Feb-27-2024"
                                LABELED_PREFIX + FILE_OF_INTEREST)


# read csv
df = pd.read_csv(LABELED_CSV_PATH, header=[1,2], index_col=[0])
label = df.iloc[:, 0].to_numpy()
visualize_mouse_gate(df=df, 
                     label=label, 
                     bodyparts=['righthindpaw', 'lefthindpaw'],
                     length_limits=(30, 60),
                     plot_N_runs=5)