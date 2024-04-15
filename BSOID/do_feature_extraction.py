# Author: Akira Kudo
# Created: 2024/03/19
# Last Updated: 2024/03/19

import os

from feature_extraction.extract_label_and_feature_from_csv import extract_label_and_feature_from_csv

THIS_FILE_DIRECTORY_ABS = os.path.dirname(os.path.abspath(__file__))

POSE = list(range(3*6)) # <- 3 (x,y,lhl) columns x 6 body parts
# 6 bodyparts: snout, rightfrontpaw, leftfrontpaw, righthindpaw, lefthindpaw, tailbase; [belly]

CLFPATH = os.path.join(THIS_FILE_DIRECTORY_ABS, 
                       "COMPUTED", "data",
                       "sav", "Leland_Feb232023", "Feb-23-2023_randomforest.sav")

SAVEPATH = os.path.join(THIS_FILE_DIRECTORY_ABS, "feature_extraction", "results")

# if running some specific files
# CSV_FOLDER_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\B-SOID\YAC128 Open Field CSVs\YAC128"
# CSV_FILES = [
#     "20230113142714_392607_m1_openfieldDLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv",
#     "20230113150304_395035_m1_openfieldDLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv",
#     "20230113153939_395035_m3_openfieldDLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
# ]

# for csv in CSV_FILES:
#     print(f"Processing file: {csv}!")
#     csvpath = os.path.join(CSV_FOLDER_PATH, csv)
#     extract_label_and_feature_from_csv(filepath=csvpath, 
#                                         pose=POSE,
#                                         clf_path=CLFPATH, 
#                                         fps=40,
#                                         brute_thresholding=False, 
#                                         threshold=0.8,
#                                         save_result=True, 
#                                         save_path=SAVEPATH,
#                                         recompute=False,  
#                                         load_path=SAVEPATH)
#         print(f"Done!")

# if running on a bunch of csvs in the given folder
ABOVE_CSV_HOLDIND_FOLDERS = os.path.abspath(
    os.path.join(__file__, "..", "..", "..", "..", "B-SOID", "YAC128 Open Field CSVs"))
CSV_HOLDING_FOLDERS = [os.path.join(ABOVE_CSV_HOLDIND_FOLDERS, "WT"),
                       os.path.join(ABOVE_CSV_HOLDIND_FOLDERS, "YAC128")]

for csv_holding_folder in CSV_HOLDING_FOLDERS:
    print(f"Processing folder: {os.path.basename(csv_holding_folder)}!")
    for filepath in os.listdir(csv_holding_folder):
        print(f"- {filepath} ... ")
        csvpath = os.path.join(csv_holding_folder, filepath)
        extract_label_and_feature_from_csv(filepath=csvpath, 
                                        pose=POSE,
                                        clf_path=CLFPATH, 
                                        fps=40,
                                        brute_thresholding=False, 
                                        threshold=0.8,
                                        save_result=True, 
                                        save_path=SAVEPATH,
                                        recompute=False,  
                                        load_path=SAVEPATH)
        print(f"Done!")