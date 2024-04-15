# Author: Akira Kudo
# Created: 2024/04/02
# Last updated: 2024/04/02

import os

from feature_analysis_and_visualization.visualization.plot_mouse_trajectory import plot_mouse_trajectory


FOLDER_OF_INTEREST = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Leland B-SOID YAC128 Analysis\YAC128\YAC128"
SAVE_PATH = os.path.join(FOLDER_OF_INTEREST, 
                         "tailbase_mouse_trajectories")


FOLDER_HOLDING_CSVS = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Leland B-SOID YAC128 Analysis\YAC128\YAC128\CSV files"
CSVS_OF_INTEREST = [os.path.join(FOLDER_HOLDING_CSVS, listed_file) 
                    for listed_file in os.listdir(FOLDER_HOLDING_CSVS)]

for csv_path in CSVS_OF_INTEREST:
    csv_name = os.path.basename(csv_path)
    plot_mouse_trajectory(csvpath=csv_path, 
                        figureName=f"{csv_name.replace('.csv','')}_figure.png", 
                        show_figure=True, 
                        save_figure=True, 
                        save_path=SAVE_PATH)