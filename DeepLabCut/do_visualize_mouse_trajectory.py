# Author: Akira Kudo
# Created: 2024/05/17
# Last Updated: 2024/07/23

import os

from dlc_io.utils import read_dlc_csv_file
from temper_with_csv_and_hdf.data_filtering.filter_based_on_boolean_array import filter_based_on_boolean_array
from temper_with_csv_and_hdf.data_filtering.identify_paw_noise import identify_bodypart_noise_by_impossible_speed
from visualization.visualize_mouse_trajectory import visualize_mouse_trajectory

BPT_ABBREV = {
    "snout" : "sn",
    "rightforepaw" : "rfp",
    "leftforepaw" : "lfp",
    "righthindpaw" : "rhp",
    "lefthindpaw" : "lhp",
    "tailbase" : "tb",
    "belly" : "bl"
}

def walk_and_generate_mouse_trajectories(walk_root : str,
                                         save_foldername : str,
                                         start : int,
                                         end : int,
                                         bodyparts : list=["tailbase"],
                                         filtered_by_impossible_speed : bool=False,
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
    :param bool filtered_by_impossible_speed: Whether to filter the rendered figures by
    using the identify_bodypart_noise_by_impossible_speed function. Defaults to False.
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

            # if specified, do some filtering based on 'impossible speed'
            if filtered_by_impossible_speed:
                raw_data = read_dlc_csv_file(csvfile)
                noise_df = identify_bodypart_noise_by_impossible_speed(
                    bpt_data=raw_data, bodyparts=bodyparts, start=0,
                    end=raw_data.shape[0], savedir=None, save_figure=False, show_figure=False,
                    print_info=False
                )

                # filter based on obtained noise info
                filtered_data = raw_data 
                for bpt in bodyparts:
                    bpt_bool_arr = noise_df[(bpt, "loc_wrng")].to_numpy()

                    filtered_data = filter_based_on_boolean_array(
                        bool_arr=bpt_bool_arr, df=filtered_data, bodyparts=[bpt], 
                        filter_mode="linear"
                    )
                data = filtered_data
            # otherwise, pass the raw csv path as data
            else:
                data = csvfile

            for bpt in bodyparts:
                if is_legitimate_bodypart(bpt):
                    # first make further subdirectories for each body part
                    bodypart_save_dir = os.path.join(traj_save_dir, BPT_ABBREV[bpt])
                    if not os.path.exists(bodypart_save_dir):
                        os.mkdir(bodypart_save_dir)
                    # do the same with the walk root folder
                    bodypart_save_walkroot_dir = os.path.join(root_save_folder, BPT_ABBREV[bpt])
                    if not os.path.exists(bodypart_save_walkroot_dir):
                        os.mkdir(bodypart_save_walkroot_dir)
                    
                    figurename = f"{file.replace('.csv', '')}_mouseTraj_{BPT_ABBREV[bpt]}.png"
                    visualize_mouse_trajectory(csv_or_df=data, 
                                            figureName=figurename, 
                                            start=start, end=end,
                                            bodypart=bpt,
                                            show_figure=show_figure,
                                            save_figure=save_figure,
                                            save_path=bodypart_save_dir)
                    # also save at the walk's root
                    visualize_mouse_trajectory(csv_or_df=data, 
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
    MOUSE_TRAJ_FOLDERNAME = "mouseTrajectory"

    END = 30*60*40 # number of frames for the 30 minutes - as the last couple minutes 
    # could include the mouse being removed from the open field, hence having weird tracking

    BODYPARTS = [
        'snout', 'tailbase', 'belly', 
        'rightforepaw', 'righthindpaw', 
        'leftforepaw', 'lefthindpaw'
        ]

    for folder in [
        r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\csv\all_2024May16_Akira"
        ]:
        walk_and_generate_mouse_trajectories(walk_root=folder,
                                            save_foldername=MOUSE_TRAJ_FOLDERNAME,
                                            start=0, end=END,
                                            bodyparts=BODYPARTS,
                                            show_figure=False,
                                            save_figure=True, 
                                            filtered_by_impossible_speed=True)