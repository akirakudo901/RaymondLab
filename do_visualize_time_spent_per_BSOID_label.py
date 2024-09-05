# Author: Akira Kudo
# Created: 2024/08/15
# Last Updated: 2024/08/30

import os

import numpy as np
import pandas as pd

from BSOID.bsoid_io.utils import read_BSOID_labeled_features
from BSOID.feature_analysis_and_visualization.behavior_groups import BehaviorGrouping
from BSOID.label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames
from DeepLabCut.visualization.visualize_data_in_scatter_dot_plot import visualize_data_in_scatter_dot_plot_from_dataframe, FILENAME, MOUSETYPE

if __name__ == "__main__":


    if True: 
        HD_GROUPNAME, WT_GROUPNAME = "YAC128", "FVB"
        HD_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\\" + \
            r"allcsv_2024_05_16_Akira\HD_filt"
        WT_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\labeled_features\\" + \
            r"allcsv_2024_05_16_Akira\WT_filt"
        FIGURE_SAVING_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\YAC128\Feb232023\figures"
    else:
        HD_GROUPNAME, WT_GROUPNAME = "Q175", "B6"
        HD_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\\" + \
            r"allcsv_2024_06_20_Akira\HD_filt"
        WT_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\labeled_features\\" + \
            r"allcsv_2024_06_20_Akira\WT_filt"
        FIGURE_SAVING_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr082024\figures"
    
    groupnames = [HD_GROUPNAME, WT_GROUPNAME]

    EXCLUDED_MICE = ["308535m1", "312153m2",# YAC HD
                     "326787m2"] # YAC WT
    FILTER_MIN_SIZE = 5

    def is_excluded_mice(path : str, mousenames : list=EXCLUDED_MICE):
        for mousename in mousenames:
            if mousename in path.replace('_', ''):
                print(f"File {path} is to be excluded as it matches the mouse name: {mousename}.")
                return True
        return False
    
    def prep_label(csv_path : str, start : int=0, end : int=30*40*60):
        """
        Preps csv by reading and extracting label, then truncating the 
        label to from start to end.
        """
        label, _ = read_BSOID_labeled_features(csv_path)
        return label[start:end+1]
    
    HD_CSVS = [os.path.join(HD_CSV_DIR, file) 
                for file in os.listdir(HD_CSV_DIR) 
                if (file.endswith(".csv") and not is_excluded_mice(file))]
    WT_CSVS = [os.path.join(WT_CSV_DIR, file) 
                for file in os.listdir(WT_CSV_DIR) 
                if (file.endswith(".csv") and not is_excluded_mice(file))]
    
    HD_LABELS = [filter_bouts_smaller_than_N_frames(label=prep_label(csv), n=FILTER_MIN_SIZE) 
                    for csv in HD_CSVS]
    WT_LABELS = [filter_bouts_smaller_than_N_frames(label=prep_label(csv), n=FILTER_MIN_SIZE) 
                    for csv in WT_CSVS]
    
    # also set up the meta labels
    NETWORK_NAME = 'Feb-23-2023'
    YAML_PATH = r'X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID\feature_analysis_and_visualization\behavior_groups_Akira_Reviewed2.yml'

    bg = BehaviorGrouping(network_name=NETWORK_NAME, yaml_path=YAML_PATH)
    groupings = bg.load_behavior_groupings(network_name=NETWORK_NAME, yaml_path=YAML_PATH)

    # convert the labels into integer meta labels
    HD_METALABELS = [convert_label_to_int_metalabel(lbl, bg) for lbl in HD_LABELS]
    WT_METALABELS = [convert_label_to_int_metalabel(lbl, bg) for lbl in WT_LABELS]

    # visualize the average time spent by mice in each 'meta label'
    if True:
        def convert_label_to_int_metalabel(label : np.ndarray, 
                                           behavior_grouping : BehaviorGrouping):
            return np.fromiter([behavior_grouping.label_to_behavioral_group_int(val) 
                                for val in label],
                                dtype=np.int32)
        
        def convert_metalabels_to_string_metalabel_counts(metalabel_array : list,
                                                          behavior_grouping : BehaviorGrouping):
            metalabel_dict = {}

            for metalabel in metalabel_array:
                unique_labels, counts = np.unique(metalabel, return_counts=True)
                
                for ul, count in zip(unique_labels, counts):
                    string_val = behavior_grouping.grouping_int_to_grouping_str[ul]
                    metalabel_counts = metalabel_dict.get(string_val, [])
                    metalabel_counts.append(count)
                    metalabel_dict[string_val] = metalabel_counts
            
            return metalabel_dict

        # convert the meta labels into meta label counts
        HD_METALABEL_COUNTS = convert_metalabels_to_string_metalabel_counts(
            HD_METALABELS, behavior_grouping=bg)
        WT_METALABEL_COUNTS = convert_metalabels_to_string_metalabel_counts(
            WT_METALABELS, behavior_grouping=bg)
        shared_metalabels = set(HD_METALABEL_COUNTS.keys()) | set(HD_METALABEL_COUNTS.keys())

        COUNTS = "counts"

        for metalabel in shared_metalabels:
            SAVENAME = f"METALABEL_framecount_mean_comparison_per_genotype_{'_'.join(groupnames)}_{metalabel}_Reviewed2.png"

            filename = ["BOGUS_m"] * (len(HD_METALABEL_COUNTS[metalabel]) + len(WT_METALABEL_COUNTS[metalabel]))
            metalabel_data = HD_METALABEL_COUNTS[metalabel] + WT_METALABEL_COUNTS[metalabel]
            mousetype = [HD_GROUPNAME] * len(HD_METALABEL_COUNTS[metalabel]) + \
                        [WT_GROUPNAME] * len(WT_METALABEL_COUNTS[metalabel])
            df = pd.DataFrame(data={FILENAME : filename,
                                    COUNTS    : metalabel_data,
                                    MOUSETYPE : mousetype})

            visualize_data_in_scatter_dot_plot_from_dataframe(
                df=df,
                y_val=COUNTS,
                xlabel='Mouse Type',
                ylabel="Frame Counts",
                title=f"Frame Counts\nPer Mouse Type\n(YAC128, Male)\n{metalabel}",
                colors=["black", "pink"],
                # sex_marker=['.', 'x'],
                sex_marker=['.', '.'],
                save_figure=True,
                save_dir=FIGURE_SAVING_PATH,
                save_name=SAVENAME,
                show_figure=False,
                show_mean=True,
                show_median=True,
                side_by_side=False
            )

    # visualize graphs for the mean bout lengths
    if True:
        for metalabel in shared_metalabels:
            SAVENAME = f"METALABEL_framecount_mean_comparison_per_genotype_{'_'.join(groupnames)}_{metalabel}_Reviewed2.png"

            filename = ["BOGUS_m"] * (len(HD_METALABEL_COUNTS[metalabel]) + len(WT_METALABEL_COUNTS[metalabel]))
            metalabel_data = HD_METALABEL_COUNTS[metalabel] + WT_METALABEL_COUNTS[metalabel]
            mousetype = [HD_GROUPNAME] * len(HD_METALABEL_COUNTS[metalabel]) + \
                        [WT_GROUPNAME] * len(WT_METALABEL_COUNTS[metalabel])
            df = pd.DataFrame(data={FILENAME : filename,
                                    COUNTS    : metalabel_data,
                                    MOUSETYPE : mousetype})

            visualize_data_in_scatter_dot_plot_from_dataframe(
                df=df,
                y_val=COUNTS,
                xlabel='Mouse Type',
                ylabel="Frame Counts",
                title=f"Frame Counts\nPer Mouse Type\n(YAC128, Male)\n{metalabel}",
                colors=["black", "pink"],
                # sex_marker=['.', 'x'],
                sex_marker=['.', '.'],
                save_figure=True,
                save_dir=FIGURE_SAVING_PATH,
                save_name=SAVENAME,
                show_figure=False,
                show_mean=True,
                show_median=True,
                side_by_side=False
            )