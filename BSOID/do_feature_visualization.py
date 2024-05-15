# Author: Akira Kudo
# Created: 2024/03/27
# Last updated: 2024/05/15

import os
from pathlib import Path

from do_feature_extraction import do_feature_extraction

from feature_analysis_and_visualization.visualization.plot_bout_length import plot_bout_length
from feature_analysis_and_visualization.visualization.plot_mouse_trajectory import plot_mouse_trajectory
from feature_analysis_and_visualization.visualization.plot_feats import plot_feats

from feature_extraction.utils import Bodypart, generate_guessed_map_of_feature_to_data_index
from feature_extraction.extract_label_and_feature_from_csv import extract_label_and_feature_from_csv
from feature_extraction.extract_pregenerated_labels_and_compute_features import extract_pregenerated_labels_and_compute_features

"""
First: creates a set of directory to organizedly store data, like so:

YOUR_ROOT
-> results
   -> figures
   -> feats_labels
-> data

*"data" will store all data we need!
"""

########################
YOUR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COMPUTED")
########################

# ACTUAL CREATION
toCreate = [os.path.join(YOUR_ROOT, "results", "figures"),
            os.path.join(YOUR_ROOT, "results", "feats_labels"),
            os.path.join(YOUR_ROOT, "data")]
for tc in toCreate:
    Path(tc).mkdir(parents=True, exist_ok=True)


###################
# BODY PARTS
###################
# Determines which body part is involved in the features generation
bodyparts = [Bodypart.SNOUT, Bodypart.RIGHTFOREPAW, Bodypart.LEFTFOREPAW,
            Bodypart.RIGHTHINDPAW, Bodypart.LEFTHINDPAW, Bodypart.TAILBASE]
"""
The POSE variable is used for extracting data from the csv.
It is a List of integers, each int corresponding to indices of columns we
  include from the csv to calculate values - but the 3rd column is indexed as
  0 in the case of DLC, which will correspond to the first bodypart column
"""
POSE = []; [POSE.extend([bp.value*3, bp.value*3+1, bp.value*3+2]) for bp in bodyparts]

# Put in the body pairs we care
relative_placement_pairs = [
    (Bodypart.RIGHTFOREPAW, Bodypart.LEFTFOREPAW),
    (Bodypart.RIGHTHINDPAW, Bodypart.LEFTHINDPAW),
    (Bodypart.SNOUT, Bodypart.TAILBASE)
]

relative_angle_pairs = [
    (Bodypart.SNOUT, Bodypart.TAILBASE),
    (Bodypart.RIGHTFOREPAW, Bodypart.TAILBASE),
    (Bodypart.RIGHTHINDPAW, Bodypart.TAILBASE),
    (Bodypart.LEFTFOREPAW, Bodypart.TAILBASE),
    (Bodypart.LEFTHINDPAW, Bodypart.TAILBASE)
]

displacement_bodyparts = [
    Bodypart.SNOUT, Bodypart.TAILBASE
]

###################
# PATHS
###################
# the path to the network folders generated
NETWORK_NAME = 'Feb-23-2023' #'Apr-08-2024'  #'Feb-26-2024'
NETWORK_SUBFOLDER =  f"Leland_{NETWORK_NAME.replace('-', '')}"
# f"Akira_{NETWORK_NAME.replace('-', '')}"

# the predictions file holds a precomputed set of labels for the FILE_OF_INTEREST file
PREDICTIONS_FILENAME = os.path.join("sav", NETWORK_SUBFOLDER,
                                    f"{NETWORK_NAME}_predictions.sav")
# the csv file of interest is the file we want to compute the feature histograms for
# if B-SOID has already ran on it, you might want to use a 'predictions.sav'
# file to extract labels from to speed up the process
CSVFILE_OF_INTEREST = [
    "312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
    
]

#"20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"
# FROM AKIRA APR08-2024 NETWORK

#     "20211018011357_242m12DLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv",
#     "20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv",
#     "20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv",
#     "20220228231804_320151_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv",
#     "20220228235946_320151_m3_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv",
#     "20230107131118_363453_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
#     ]
# FROM AKIRA FEB26-2024 NETWORK

# FROM LELAND FEB23-2023 NETWORK
# r"312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
# r"20211018011357_242m12DLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
# r"20230107131118_363453_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"

PREDICTIONS_PATH = os.path.join(YOUR_ROOT, "data", PREDICTIONS_FILENAME)
CSVFOLDER_PATH = os.path.join(YOUR_ROOT, "data", "csv", NETWORK_SUBFOLDER)
# this holds the path to the classifier used to generate all these data; set it to the correct one,
# even if only utilizing the 'extract_pregenerated_labels_and_compute_features' functionality
CLF_FILENAME = f"{NETWORK_NAME}_randomforest.sav"
# r"Feb-23-2023_randomforest.sav"
CLF_SAV_PATH = os.path.join(YOUR_ROOT, "data", "sav", NETWORK_SUBFOLDER, CLF_FILENAME)

# prob changes less; first is where figures are saved, and second is where
# computed features & labels are saved for later speedup
FIGURE_SAVING_PATH = os.path.join(YOUR_ROOT, "results", "figures", NETWORK_SUBFOLDER)
COMPUTED_FEATURE_SAVING_PATH = os.path.join(YOUR_ROOT, "results", "feats_labels", NETWORK_SUBFOLDER)


######################
# OTHER CONFIGURATIONS
######################
# whether to show the figure - if on a computer, might want to check afterwards
SHOW_FIGURE = False
# whether to save rendered histogram figures
SAVE_FIGURE = True
# whether to use a log scale for for the y-ticks of the histogram
LOGSCALE = True
# whether to use adaptive or brute thresholding for filtering values
# when runnign the adp_filt function - check its definition for more
BRUTE_THRESHOLDING = False
# the brute threshold to use if BRUTE_THRESHOLDING is True
BRUTE_THRESHOLDING_THRESH = 0.8
# This is the list of labels to show
GROUPS_TO_SHOW = [32,38]
# fps of input videos
FPS = 40


# MAIN FUNCTION
def main(csvfile : str):
    """
    Plots features as analyzed and output by B-SOID.
    1) Takes in an already analyzed csv path together with its
       analysis-resulting predictions.
    2) Extracts labels out of the csv.
    3) Extracts kinematic features from the csv.
    4) Plots the results, leveraging the labels & kinematics extracted.

    :param str csvfile: The path to the csv file we analyze.
    """
    csvfullpath = os.path.join(CSVFOLDER_PATH, csvfile)

    labels, features = do_feature_extraction(
        csv_path=csvfullpath,
        predictions_path=PREDICTIONS_PATH,
        clf_sav_path=CLF_SAV_PATH,
        computed_feature_saving_path=COMPUTED_FEATURE_SAVING_PATH,
        fps=FPS,
        brute_thresholding=BRUTE_THRESHOLDING, 
        threshold=BRUTE_THRESHOLDING_THRESH,
        recompute=False,
        save_result=True,
        pose=POSE
        )

    # generate map of features to guessed index
    featname_to_idx_map = generate_guessed_map_of_feature_to_data_index(
        bodyparts
    )
    # show the unfiltered mouse trajectory to check for gitter
    plot_mouse_trajectory(csvpath=csvfullpath,
                            figureName=os.path.basename(csvfullpath).replace('.csv', ''),
                            start=0, end=None, bodypart="tailbase",
                            show_figure=SHOW_FIGURE,
                            save_figure=SAVE_FIGURE,
                            save_path=os.path.join(FIGURE_SAVING_PATH,
                                                "mouseTrajectory"))

    # plot the features
    plot_feats(features, labels,
                GROUPS_TO_SHOW,
                relative_placement_pairs, relative_angle_pairs, displacement_bodyparts,
                feature_to_index_map=featname_to_idx_map,
                figure_save_dir=FIGURE_SAVING_PATH,
                csv_name=csvfile,
                show_figure=SHOW_FIGURE, 
                save_figure=SAVE_FIGURE,
                use_logscale=LOGSCALE,
                brute_thresholding=BRUTE_THRESHOLDING)

    plot_bout_length(labels,
                    csv_name=csvfile,
                    figure_save_dir=FIGURE_SAVING_PATH,
                    show_figure=SHOW_FIGURE,
                    save_figure=SAVE_FIGURE,
                    use_logscale=LOGSCALE)
    

def do_feature_visualization(csvfile: str, 
                            csvfolder_path: str, 
                            clf_sav_path: str, 
                            predictions_path : str,
                            computed_feature_saving_path: str,
                            figure_saving_path: str, 
                            bodyparts: list,
                            groups_to_show: list, 
                            relative_placement_pairs: list,
                            relative_angle_pairs: list, 
                            displacement_bodyparts: list, 
                            show_figure: bool, 
                            save_figure: bool,
                            logscale: bool, 
                            brute_thresholding: bool, 
                            brute_threshold : float=0.8):
    """
    Plots features as analyzed and output by B-SOID.
    1) Takes in an already analyzed csv path together with its
       analysis-resulting predictions.
    2) Extracts labels out of the csv.
    3) Extracts kinematic features from the csv.
    4) Plots the results, leveraging the labels & kinematics extracted.

    :param str csvfile: The path to the csv file we analyze.
    :param str csvfolder_path: The path to the folder containing the csv file.
    :param str clf_sav_path: The path to the classifier saving location.
    :param str predictions_path: The path to previous predictions.
    :param str computed_feature_saving_path: The path to save computed features.
    :param str figure_saving_path: The path to save generated figures.
    :param list bodyparts: List of body parts for feature extraction.
    :param list groups_to_show: Groups of features to show in plots.
    :param list relative_placement_pairs: Pairs of body parts for relative placement.
    :param list relative_angle_pairs: Pairs of body parts for relative angle calculations.
    :param list displacement_bodyparts: Body parts for displacement calculations.
    :param bool show_figure: Whether to show generated figures.
    :param bool save_figure: Whether to save generated figures.
    :param bool logscale: Whether to use log scale in plots.
    :param bool brute_thresholding: Whether to use brute thresholding in plots.
    :param float brute_threshold: Threshold for brute thresholding, defaults to 0.8.
    """

    # create non-existent folders if needed
    Path(computed_feature_saving_path).mkdir(parents=True, exist_ok=True)
    Path(figure_saving_path).mkdir(parents=True, exist_ok=True)
    
    csvfullpath = os.path.join(csvfolder_path, csvfile)

    labels, features = do_feature_extraction(
        csv_path=csvfullpath,
        predictions_path=predictions_path,
        clf_sav_path=clf_sav_path,
        computed_feature_saving_path=computed_feature_saving_path,
        fps=FPS,
        brute_thresholding=brute_thresholding, 
        threshold=brute_threshold,
        recompute=False,
        save_result=True,
        pose=POSE
        )
    
    # generate map of features to guessed index
    featname_to_idx_map = generate_guessed_map_of_feature_to_data_index(
        bodyparts
    )
    # show the unfiltered mouse trajectory to check for gitter
    plot_mouse_trajectory(csvpath=csvfullpath,
                          figureName=os.path.basename(csvfullpath).replace('.csv', ''),
                          start=0, end=None, bodypart="tailbase",
                          show_figure=show_figure,
                          save_figure=save_figure,
                          save_path=os.path.join(figure_saving_path, "mouseTrajectory"))

    # plot the features
    plot_feats(features, labels,
                groups_to_show,
                relative_placement_pairs, relative_angle_pairs, displacement_bodyparts,
                feature_to_index_map=featname_to_idx_map,
                figure_save_dir=figure_saving_path,
                csv_name=csvfile,
                show_figure=show_figure, 
                save_figure=save_figure,
                use_logscale=logscale,
                brute_thresholding=brute_thresholding)

    plot_bout_length(labels,
                    csv_name=csvfile,
                    figure_save_dir=figure_saving_path,
                    show_figure=show_figure,
                    save_figure=save_figure,
                    use_logscale=logscale)


if __name__ == "__main__":
    # from feature_analysis_and_visualization.behavior_groups import BehaviorGrouping
    
    # LABEL_OF_INTEREST = "WallRearing"
    
    # GROUPS_TO_SHOW = []

    # bg = BehaviorGrouping(network_name=NETWORK_NAME)
    # for behavior_group, labels in bg.groupings_str.items():
    #     if behavior_group == LABEL_OF_INTEREST:
    #         GROUPS_TO_SHOW = labels
    
    # if len(GROUPS_TO_SHOW) == 0: 
    #     raise Exception(f"We couldn't find the behavior group {LABEL_OF_INTEREST}, " +
    #                     "or no label seem to belong to it...")

    # automatically execute compute on every csv in the specified folder 
    # CSVFILE_OF_INTEREST = os.listdir(CSVFOLDER_PATH)

    for csvfile in CSVFILE_OF_INTEREST:
        main(csvfile)
