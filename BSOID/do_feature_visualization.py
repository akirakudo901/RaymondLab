# Author: Akira Kudo
# Created: 2024/03/27
# Last updated: 2024/03/27

import os
from pathlib import Path

from feature_analysis_and_visualization.visualization.plot_bout_length import plot_bout_length
from feature_analysis_and_visualization.visualization.plot_mouse_trajectory import plot_mouse_trajectory
from feature_analysis_and_visualization.visualization.plot_feats import plot_feats

from feature_extraction.utils import Bodypart, generate_guessed_map_of_feature_to_data_index
from feature_extraction.extract_label_and_feature_from_csv import extract_label_and_feature_from_csv

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
YOUR_ROOT = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID\COMPUTED"
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
    (Bodypart.SNOUT, Bodypart.RIGHTFOREPAW),
    (Bodypart.SNOUT, Bodypart.RIGHTHINDPAW),
    (Bodypart.RIGHTFOREPAW, Bodypart.LEFTFOREPAW),
    (Bodypart.SNOUT, Bodypart.TAILBASE)
]

relative_angle_pairs = [
    (Bodypart.SNOUT, Bodypart.TAILBASE)
]

displacement_bodyparts = [
    Bodypart.SNOUT, Bodypart.TAILBASE
]

###################
# PATHS
###################
# the path to the network folders generated
NETWORK_NAME = 'Feb-23-2023' #'Feb-26-2024'
NETWORK_SUBFOLDER = f"Leland_{NETWORK_NAME.replace('-', '')}"
# f"Akira_{NETWORK_NAME.replace('-', '')}"

# the predictions file holds a precomputed set of labels for the FILE_OF_INTEREST file
PREDICTIONS_FILENAME = os.path.join("sav", NETWORK_SUBFOLDER,
                                    f"{NETWORK_NAME}_predictions.sav")
# the csv file of interest is the file we want to compute the feature histograms for
# if B-SOID has already ran on it, you might want to use a 'predictions.sav'
# file to extract labels from to speed up the process
CSVFILE_OF_INTEREST = []
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
BRUTE_THRESHOLDING = True
# This is the list of labels to show
GROUPS_TO_SHOW = []# list(range(0, 38)) + list(range(39, 45))


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

    labels, features = None, None
    print("Computing both labels and features from csv!")
    labels, features = extract_label_and_feature_from_csv(
        filepath=csvfullpath, pose=POSE, clf_path=CLF_SAV_PATH, fps=40,
        save_result=True, save_path=COMPUTED_FEATURE_SAVING_PATH,
        recompute=False,  load_path=COMPUTED_FEATURE_SAVING_PATH)
    print("End.")

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

if __name__ == "__main__":
    from feature_analysis_and_visualization.behavior_groups import BehaviorGrouping
    
    LABEL_OF_INTEREST = "WallRearing"
    
    GROUPS_TO_SHOW = []

    bg = BehaviorGrouping(network_name=NETWORK_NAME)
    for behavior_group, labels in bg.groupings.items():
        if behavior_group == LABEL_OF_INTEREST:
            GROUPS_TO_SHOW = labels
    
    if len(GROUPS_TO_SHOW) == 0: 
        raise Exception(f"We couldn't find the behavior group {LABEL_OF_INTEREST}, " +
                        "or no label seem to belong to it...")

    # automatically execute compute on every csv in the specified folder 
    CSVFILE_OF_INTEREST = os.listdir(CSVFOLDER_PATH)
    for csvfile in CSVFILE_OF_INTEREST:
        main(csvfile)