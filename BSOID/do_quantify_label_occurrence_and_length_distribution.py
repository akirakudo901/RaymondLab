# Author: Akira Kudo
# Created: 2024/04/04
# Last Updated: 2024/05/29

import os

import numpy as np

from feature_analysis_and_visualization.visualization.quantify_labels import quantify_label_occurrence_and_length_distribution, visualize_label_occurrences_heatmaps, visualize_group_average_label_occurrences
from bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
from feature_analysis_and_visualization.utils import get_mousename

if False:
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


if True:
    import os
    # SETTING
    MOUSETYPE = "Q175-B6" if False else "YAC128-FVB"
    SEPEARATE_WT_AND_HD = True

    Q175_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\allcsv_2024_05_16_Akira"
    YAC_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\allcsv_2024_05_16_Akira"
    
    Q175_SAVE_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr082024\figures"
    YAC_SAVE_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\Feb232023\figures"

    # EXECUTION
    save_folder = Q175_SAVE_FOLDER if MOUSETYPE == "Q175-B6" else YAC_SAVE_FOLDER
    
    if SEPEARATE_WT_AND_HD:
        CSV_FOLDER_LIST = [
            os.path.join(Q175_FOLDER, "HD_filt"),
            os.path.join(Q175_FOLDER, "WT_filt"),
            ] if MOUSETYPE == "Q175-B6" else [
            os.path.join(YAC_FOLDER, "HD_filt"),
            os.path.join(YAC_FOLDER, "WT_filt"),
            ]
        
        GROUPNAMES = [
            "Q175", "B6"
            ] if MOUSETYPE == "Q175-B6" else [
                "YAC128", "FVB"
            ]
        
        CSV_PATH_LIST = [
            [os.path.join(folder, file) for file in os.listdir(folder)] 
                for folder in CSV_FOLDER_LIST
        ]
    else:
        CSV_FOLDER_LIST = [
            os.path.join(Q175_FOLDER, "HD_filt"),
            os.path.join(Q175_FOLDER, "WT_filt"),
            ] if MOUSETYPE == "Q175-B6" else [
            os.path.join(YAC_FOLDER, "HD_filt"),
            os.path.join(YAC_FOLDER, "WT_filt"),
            ]
        
        GROUPNAMES = ["Q175-B6"] if MOUSETYPE == "Q175-B6" else ["YAC128-FVB"]
        
        CSV_PATH_LIST = [[
            os.path.join(folder, file) for folder in CSV_FOLDER_LIST for file in os.listdir(folder)
            ]]

    LABEL_LIST = [
        [read_BSOID_labeled_features(csv)[0] for csv in group]
        for group in CSV_PATH_LIST
    ]
    MOUSENAMES = [
        [get_mousename(path) for path in csvpath_group] 
            for csvpath_group in CSV_PATH_LIST
    ]

    if True:
        visualize_label_occurrences_heatmaps(
            group_of_labels=LABEL_LIST,
            group_names=GROUPNAMES,
            mousenames=MOUSENAMES,
            labels_to_check=None,
            ylabel="Individuals",
            save_dir=save_folder,
            save_name=f"behavior_label_occurrence_{MOUSETYPE}{'separate' if SEPEARATE_WT_AND_HD else ''}_vert",
            vmin=None,
            vmax=None,
            xlabel="Label Groups",
            title=f"Behavior Label Occurrence (%) for {MOUSETYPE}",
            save_figure=True,
            show_figure=True,
            figsize=(12,6)
        )

    if False:
        visualize_group_average_label_occurrences(
            group_of_labels=LABEL_LIST,
            group_names=GROUPNAMES,
            mousenames=MOUSENAMES,
            labels_to_check=None,
            ylabel="Individuals",
            save_dir=save_folder,
            save_name=f"behavior_label_occurrence_{MOUSETYPE}{'separate' if SEPEARATE_WT_AND_HD else ''}",
            xlabel="Label Groups",
            title=f"Behavior Label Occurrence (%) for {MOUSETYPE}",
            save_figure=False,
            show_figure=True,
            figsize=(12,6)
            )