# Author: Akira Kudo
# Created: 2024/03/19
# Last Updated: 2024/03/19

import os

from feature_extraction.extract_label_and_feature_from_csv import extract_label_and_feature_from_csv

POSE = list(range(21)) # <- 3 (x,y,lhl) columns x 6 body parts = 21 columns
# 6 bodyparts: snout, rightfrontpaw, leftfrontpaw, righthindpaw, lefthindpaw, tailbase; [belly]

CLFPATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\feature_visualization_JupyterNotebook\data" + \
          r"\sav\Leland_Feb232023_network\Feb-23-2023_randomforest.sav"

SAVEPATH = "./feature_extraction/results"

ABOVE_CSV_HOLDIND_FOLDERS = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\B-SOID\YAC128 Open Field CSVs"
CSV_HOLDING_FOLDERS = [os.path.join(ABOVE_CSV_HOLDIND_FOLDERS, "WT"),
                       os.path.join(ABOVE_CSV_HOLDIND_FOLDERS, "YAC128")]

for csv_holding_folder in CSV_HOLDING_FOLDERS:
    print(f"Processing folder: {os.path.basename(csv_holding_folder)}!")
    for filepath in os.listdir(csv_holding_folder):
        print(f"- {filepath} ... ", end="")
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