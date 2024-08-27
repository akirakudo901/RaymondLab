# Author: Akira Kudo
# Created: 2024/04/04
# Last Updated: 2024/08/23

import os

import numpy as np

from feature_analysis_and_visualization.visualization.quantify_labels import quantify_label_occurrence_and_length_distribution, quantify_max_percentage_per_mouse_per_labels, visualize_label_occurrences_heatmaps, visualize_group_average_label_occurrences
from bsoid_io.utils import read_BSOID_labeled_csv, read_BSOID_labeled_features
from feature_analysis_and_visualization.behavior_groups import BehaviorGrouping
from feature_analysis_and_visualization.utils import get_mousename
from label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

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

    EXCLUDED_MICE = ["308535m1", "312153m2"]

    def is_excluded_mice(path : str, mousenames : list=EXCLUDED_MICE):
        for mousename in mousenames:
            if mousename in path.replace('_', ''):
                print(f"File {path} is to be excluded as it matches the mouse name: {mousename}.")
                return True
        return False

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
            [os.path.join(folder, file) for file in os.listdir(folder)
             if (file.endswith(".csv") and not is_excluded_mice(file))] 
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
        
        CSV_PATH_LIST = [
            [os.path.join(folder, file) for file in os.listdir(folder)
             if (file.endswith(".csv") and not is_excluded_mice(file))]
            for folder in CSV_FOLDER_LIST 
            ]
        
    def prep_label(csv_path : str, start : int=0, end : int=30*40*60):
        """
        Preps csv by reading and extracting label, then truncating the 
        label to from start to end.
        """
        label, _ = read_BSOID_labeled_features(csv_path)
        return label[start:end+1]

    LABEL_LIST = [
        [prep_label(csv) for csv in group]
        for group in CSV_PATH_LIST
    ]

    MOUSENAMES = [
        [get_mousename(path) for path in csvpath_group] 
            for csvpath_group in CSV_PATH_LIST
    ]

    if True:
        def convert_label_to_int_metalabel(label : np.ndarray, 
                                           behavior_grouping : BehaviorGrouping):
            return np.fromiter([behavior_grouping.label_to_behavioral_group_int(val) 
                                for val in label],
                                dtype=np.int32)

        NETWORK_NAME = 'Feb-23-2023'
        YAML_PATH = r'X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID\feature_analysis_and_visualization\behavior_groups_Akira_Reviewed.yml'
        FILTER_MIN_SIZE = 5

        LABEL_LIST = [
            [filter_bouts_smaller_than_N_frames(label, n=FILTER_MIN_SIZE) for label in group]
            for group in LABEL_LIST
        ]

        bg = BehaviorGrouping(network_name=NETWORK_NAME, yaml_path=YAML_PATH)
        groupings = bg.load_behavior_groupings(network_name=NETWORK_NAME, yaml_path=YAML_PATH)

        # convert the labels into integer meta labels
        METALABEL_LIST = [
            [convert_label_to_int_metalabel(lbl, bg) for lbl in group]
            for group in LABEL_LIST
            ]
        
        # visualize occurrence of meta label between genotypes
        if True:
            quantify_label_occurrence_and_length_distribution(
                group_of_labels=METALABEL_LIST,
                group_names=GROUPNAMES, 
                use_logscale=False,
                save_dir=save_folder,
                save_name=f"Metalabel_Occurrence_{MOUSETYPE}{'separate' if SEPEARATE_WT_AND_HD else ''}_Reviewed.png",
                label_to_name=bg.grouping_int_to_grouping_str,
                save_figure=True,
                show_figure=False
            )

        # visualize occurrence of meta label for each mouse
        if False:
            each_mouse_folder = "eachMouse_META_Reviewed"
            save_folder = os.path.join(save_folder, each_mouse_folder)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            for metalabel_group, mousename_group in zip(METALABEL_LIST, MOUSENAMES):
                for metalabel, mousename in zip(metalabel_group, mousename_group):
                    quantify_label_occurrence_and_length_distribution(
                            group_of_labels=[[metalabel,],],
                            group_names=[mousename,],
                            use_logscale=False,
                            save_dir=save_folder,
                            save_name=f"Metalabel_Occurrence_{mousename}_Reviewed.png",
                            label_to_name=bg.grouping_int_to_grouping_str,
                            save_figure=True,
                            show_figure=False
                        )
        
        # visualize occurrence of the original labels for each mouse
        if False:
            each_mouse_folder = "eachMouse_Orig_Reviewed"
            save_folder = os.path.join(save_folder, each_mouse_folder)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            for label_group, mousename_group in zip(LABEL_LIST, MOUSENAMES):
                for label, mousename in zip(label_group, mousename_group):
                    quantify_label_occurrence_and_length_distribution(
                            group_of_labels=[[label,],],
                            group_names=[mousename,],
                            use_logscale=False,
                            save_dir=save_folder,
                            save_name=f"Label_Occurrence_{mousename}_reviewed.png",
                            save_figure=True,
                            show_figure=False
                        )

    if False:
        visualize_label_occurrences_heatmaps(
            group_of_labels=LABEL_LIST,
            group_names=GROUPNAMES,
            mousenames=MOUSENAMES,
            labels_to_check=None,
            ylabel="Individuals",
            display_threshold=0,
            save_dir=save_folder,
            save_name=f"behavior_label_occurrence_{MOUSETYPE}{'separate' if SEPEARATE_WT_AND_HD else ''}_vert_reviewed",
            vmin=None,
            vmax=None,
            xlabel="Label Groups",
            title=f"Behavior Label Occurrence (%) for {MOUSETYPE}",
            save_figure=True,
            show_figure=False,
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
    
    # visualize what the maximum percentage of each mouse labels each label covers
    if False:
        SAVEDIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\Feb232023\figures"
        quantify_max_percentage_per_mouse_per_labels(labels=[lbl for group in LABEL_LIST for lbl in group],
                                                     mousenames=[msnm for group in MOUSENAMES for msnm in group], 
                                                     label_groups=None,
                                                     savedir=SAVEDIR,
                                                     savename="MaxAndTotalPercentageOfLabelPerGroup_YAC128_Reviewed.txt",
                                                     show_message=True,
                                                     save_result=True)