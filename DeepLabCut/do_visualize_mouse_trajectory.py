# Author: Akira Kudo
# Created: 2024/05/17
# Last Updated: 2024/06/26

import os

from visualization.visualize_mouse_trajectory import visualize_mouse_trajectory

def walk_and_generate_mouse_trajectories(walk_root : str,
                                         save_foldername : str,
                                         start : int,
                                         end : int,
                                         bodyparts : list=["tailbase"],
                                         show_figure : bool=True,
                                         save_figure : bool=True):
    """
    Walks down the subdirectories from the specified root, 
    generating mouse trajectory plots for specified body parts
    between frames start and end. Plots are saved both at:
    1) the location where the csv was found,
    2) the root of the walk given
    in a folder named 'save_foldername'.

    :param str walk_root: Root from which we start the walk.
    :param str save_foldername: Name of the folder where we save the plots.
    Will be created newly if it doesn't exist.
    :param int start: Starting frame for generating the plot.
    :param int end: End frame for generating the plot.
    :param list bodyparts: A list of body parts for which the plot will be 
    created, one for each legitimate body part. Those that aren't legitimate
    are skipped.
    :param bool show_figure: Whether to show figures, defaults to True
    :param bool save_figure: Whether to save figures, defaults to True
    """
    # first create the 'save_foldername' folder in the walk's root
    root_save_folder = os.path.join(walk_root, save_foldername)
    if not os.path.exists(root_save_folder):
        os.mkdir(root_save_folder)

    # if we want to save the result in the same csvs, with the same folders 
    # in a given os structure
    for root, _, filenames in os.walk(walk_root):
        # if there is at least one csv in a given directory
        if len([f for f in filenames if '.csv' in f]) > 0:
            traj_save_dir = os.path.join(root, save_foldername)
            if not os.path.exists(traj_save_dir):
                os.mkdir(traj_save_dir)
        else:
            continue

        print(f"Processing: {root}!")
        for file in filenames:
            csvfile = os.path.join(root, file)

            for bpt in bodyparts:
                if is_legitimate_bodypart(bpt):
                    # first make further subdirectories for each body part
                    bodypart_save_dir = os.path.join(traj_save_dir, bpt)
                    if not os.path.exists(bodypart_save_dir):
                        os.mkdir(bodypart_save_dir)
                    # do the same with the walk root folder
                    bodypart_save_walkroot_dir = os.path.join(root_save_folder, bpt)
                    if not os.path.exists(bodypart_save_walkroot_dir):
                        os.mkdir(bodypart_save_walkroot_dir)

                    figurename = f"{file.replace('.csv', '')}_mouseTraj_{bpt}.png"
                    visualize_mouse_trajectory(csvpath=csvfile, 
                                            figureName=figurename, 
                                            start=start, end=end,
                                            bodypart=bpt,
                                            show_figure=show_figure,
                                            save_figure=save_figure,
                                            save_path=bodypart_save_dir)
                    # also save at the walk's root
                    visualize_mouse_trajectory(csvpath=csvfile, 
                                            figureName=figurename, 
                                            start=start, end=end,
                                            bodypart=bpt,
                                            show_figure=show_figure,
                                            save_figure=save_figure,
                                            save_path=bodypart_save_walkroot_dir)

def is_legitimate_bodypart(bpt : str):
    return bpt in ['snout', 'rightforepaw', 'leftforepaw', 
                   'righthindpaw', 'lefthindpaw', 'tailbase',
                   'belly']

if __name__ == "__main__":
    TOFILL_CSV_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\{}\csv\allcsv_2024_05_16_Akira"

    MOUSE_TRAJ_FOLDERNAME = "mouseTrajectory"

    END = 30*60*40 # number of frames for the 30 minutes - as the last couple minutes 
    # could include the mouse being removed from the open field, hence having weird tracking

    BODYPARTS = ['snout', 'tailbase', 'belly']

    for folder in [
        # TOFILL_CSV_DIR.format("Q175"), 
        # TOFILL_CSV_DIR.format("YAC128"),
        # r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\black\it0-2000k"
        r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\csv\allcsv_2024_06_20_Akira"
        ]:
        walk_and_generate_mouse_trajectories(walk_root=folder,
                                            save_foldername=MOUSE_TRAJ_FOLDERNAME,
                                            start=0, end=END,
                                            bodyparts=BODYPARTS,
                                            show_figure=False,
                                            save_figure=True)