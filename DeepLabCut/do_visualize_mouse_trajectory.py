# Author: Akira Kudo
# Created: 2024/05/17
# Last Updated: 2024/05/21

import os

from visualization.visualize_mouse_trajectory import visualize_mouse_trajectory

CSV_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv"

MOUSE_TRAJECTORY_SAVE_DIR = CSV_DIR
# r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\results\figures\Akira_Apr082024\mouseTrajectory"

MOUSE_TRAJ_FOLDERNAME = "mouseTrajectory"

# if we want to save the result in the same csvs, with the same folders in a given os structure
for root, dirnames, filenames in os.walk(CSV_DIR):
    # if there is at least one csv in a given directory
    if len([f for f in filenames if '.csv' in f]) > 0:
        MOUSE_TRAJECTORY_SAVE_DIR = os.path.join(root, MOUSE_TRAJ_FOLDERNAME)
        if not os.path.exists(MOUSE_TRAJECTORY_SAVE_DIR):
            os.mkdir(MOUSE_TRAJECTORY_SAVE_DIR)
    else:
        continue

    print(f"Processing: {root}!")
    for file in filenames:
        csvfile = os.path.join(root, file)

        figurename = f"{file.replace('.csv', '')}_mouseTraj.png"

        visualize_mouse_trajectory(csvpath=csvfile, 
                                   figureName=figurename, 
                                   show_figure=False, 
                                   save_figure=True, 
                                   save_path=MOUSE_TRAJECTORY_SAVE_DIR)