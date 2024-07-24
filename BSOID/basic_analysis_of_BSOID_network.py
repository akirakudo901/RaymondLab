# Author: Akira Kudo
# Created: 2024/04/17
# Last Updated: 2024/05/15

"""
This script attempts to have a basic analysis of a new B-SOID network in one place.
It will:
- Generate feature distributions over different behavior groups.
- Generate behavior snippets of the top N lengthy groups post filtering.
- Generate figures for label occurrence & behavior bouts occurrence post filtering.
- TO COME! Generate a preliminary assessment of which group belong with which other??
"""

import os
from pathlib import Path
from typing import List, Tuple

from bsoid_io.utils import read_BSOID_labeled_csv
from do_feature_visualization import do_feature_visualization
from feature_analysis_and_visualization.utils import get_mousename
from feature_analysis_and_visualization.visualization.quantify_labels import quantify_label_occurrence_and_length_distribution
from feature_extraction.utils import Bodypart
from label_behavior_bits.create_labeled_behavior_bits import create_labeled_behavior_bits, extract_label_from_labeled_csv, filter_bouts_smaller_than_N_frames

RESULT_FOLDER = "Result"

###############################################
# TO BE CHANGED AS NEEDED : DEFAULT ARGUMENTS #
###############################################

###################
# BODY PARTS
###################
# Determines which body part is involved in the features generation
BODYPARTS = [Bodypart.SNOUT,        Bodypart.RIGHTFOREPAW, Bodypart.LEFTFOREPAW,
             Bodypart.RIGHTHINDPAW, Bodypart.LEFTHINDPAW,  Bodypart.TAILBASE]

# we exclude "belly" as it isn't used to classify in this B-SOID
BODYPARTS_STR = ["snout", "rightforepaw", "leftforepaw", "righthindpaw", "lefthindpaw",  "tailbase"] 

# Put in the body pairs we care
RELATIVE_PLACEMENT_PAIRS = [
    (Bodypart.RIGHTFOREPAW, Bodypart.LEFTFOREPAW),
    (Bodypart.RIGHTHINDPAW, Bodypart.LEFTHINDPAW),
    (Bodypart.SNOUT, Bodypart.TAILBASE)
]

RELATIVE_ANGLE_PAIRS = [
    (Bodypart.SNOUT, Bodypart.TAILBASE),
    (Bodypart.RIGHTFOREPAW, Bodypart.TAILBASE),
    (Bodypart.RIGHTHINDPAW, Bodypart.TAILBASE),
    (Bodypart.LEFTFOREPAW, Bodypart.TAILBASE),
    (Bodypart.LEFTHINDPAW, Bodypart.TAILBASE)
]

DISPLACEMENT_BODYPARTS = [
    Bodypart.SNOUT, Bodypart.TAILBASE
]

# ###################
# # PATHS
# ###################
# # the path to the network folders generated
# NETWORK_SUBFOLDER = f'Akira_{NETWORK_NAME.replace("-", "")}'

# # the csv file of interest is the file we want to compute the feature histograms for
# # they are held in CSVFOLDER_PATH
# CSVFILE_OF_INTEREST = [file for file in os.listdir(CSVFOLDER_PATH) if file.endswith('.csv')]

# # this holds the path to the classifier used to generate all these data; set it to the correct one,
# # even if only utilizing the 'extract_pregenerated_labels_and_compute_features' functionality
# CLF_FILENAME = f"{NETWORK_NAME}_randomforest.sav"
# # r"Feb-23-2023_randomforest.sav"
# CLF_SAV_PATH = os.path.join(CLF_SAV_FOLDER, CLF_FILENAME)

# # prob changes less; first is where figures are saved, and second is where
# # computed features & labels are saved for later speedup
# SAVING_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COMPUTED")
# FIGURE_SAVING_PATH = os.path.join(SAVING_FOLDER, "results", "figures", NETWORK_SUBFOLDER)
# COMPUTED_FEATURE_SAVING_PATH = os.path.join(SAVING_FOLDER, "results", "feats_labels", NETWORK_SUBFOLDER)

# ######################
# # OTHER CONFIGURATIONS
# ######################
# # whether to show the figure - if on a computer, might want to check afterwards
# SHOW_FIGURE = False
# # whether to save rendered histogram figures
# SAVE_FIGURE = True
# # whether to use a log scale for for the y-ticks of the histogram
# LOGSCALE = True
# # whether to use adaptive or brute thresholding for filtering values
# # when runnign the adp_filt function - check its definition for more
# BRUTE_THRESHOLDING = False
# # This is the list of labels to show
# GROUPS_TO_SHOW = [] #0,1,5,9,10,12,14,15,17,22,24,28,29,30


# #########################################################################################################################
# # GENERATE BEHAVIOR SNIPPETS

# FPS = 40
# FILTERING_NOISE_MAX_LENGTH = 5 # max length of noise filtered via filter_bouts_smaller_than_N_frames
# MIN_DESIRED_BOUT_LENGTH = 500 # if generating at random, find bouts longer than this
# COUNTS = 5 # how many videos we generate
# OUTPUT_FPS = 40
# TRAILPOINTS = 0
# TOP_OR_RANDOM = "top" #"random"

# DOTSIZE = 7
# COLORMAP = "rainbow" # obtained from config.yaml on DLC side

# #####DEFINING FILE OF INTEREST#####
# csv_from_video_path = os.path.basename(VIDEO_PATH.replace('.mp4', ''))
# FILE_OF_INTEREST = next((filename for filename in CSVFILE_OF_INTEREST 
#                          if csv_from_video_path in filename), None) # choose first csv at random
# if FILE_OF_INTEREST is None:
#     raise Exception("Variable VIDEO_PATH has to be set such that it matches at least " + 
#                     "one CSV in the folder given by CSVFOLDER_PATH; but failed... \n" + 
#                     f"not-found video file: {csv_from_video_path}.")

# #####DEFINING OUTPUT FOLDER#####
# OUTPUT_FOLDER = os.path.join(SAVING_FOLDER, 'results', 'videos', NETWORK_SUBFOLDER, 'bits')

# #####DEFINING OTHER IMPORTANT FOLDERS#####
# FRAME_DIR = os.path.join(SAVING_FOLDER, 'results', 'pngs', get_mousename(FILE_OF_INTEREST))

# DATA_CSV_PATH = os.path.join(CSVFOLDER_PATH, FILE_OF_INTEREST)

# LABELED_CSV_PATH = os.path.join(LABELED_CSV_FOLDER, 
#                                 [file for file in os.listdir(LABELED_CSV_FOLDER) 
#                                  if FILE_OF_INTEREST in str(file)][0])

# OUTPUT_PATH  = os.path.join(OUTPUT_FOLDER, get_mousename(FILE_OF_INTEREST))

# # GENERATE THE IMPORTANT FOLDERS IF NOT THERE
# toCreate = [FRAME_DIR, OUTPUT_FOLDER, FIGURE_SAVING_PATH, COMPUTED_FEATURE_SAVING_PATH]
# for tc in toCreate:
#     Path(tc).mkdir(parents=True, exist_ok=True)




# #########################################################################################################################
# # EXECUTION!!!
# ##################

# # QUANTIFY BEHAVIOR SNIPPETS
# def convert_folder_to_list_of_label(folder : str):
#     csvs = [os.path.join(folder, file) for file in os.listdir(folder) 
#             if '.csv' in file]
#     labels = [read_BSOID_labeled_csv(csv)[0] for csv in csvs]
#     return labels

# labels = convert_folder_to_list_of_label(LABELED_CSV_FOLDER)

# quantify_label_occurrence_and_length_distribution(
#     group_of_labels=[labels],
#     group_names=[GROUP_NAME], 
#     save_dir=FIGURE_SAVING_PATH, 
#     save_name=f"{GROUP_NAME}.png",
#     use_logscale=False,
#     save_figure=SAVE_FIGURE
# )

# # THEN GENERATE BEHAVIOR BITS
# print("Generating behavior bits...")

# labels = extract_label_from_labeled_csv(LABELED_CSV_PATH)

# print("Pre-filtering...")
# filtered = filter_bouts_smaller_than_N_frames(labels, n=FILTERING_NOISE_MAX_LENGTH)
# print("Done!")

# create_labeled_behavior_bits(labels=filtered, 
#                             crit=MIN_DESIRED_BOUT_LENGTH / FPS, 
#                             counts=COUNTS,
#                             output_fps=OUTPUT_FPS, 
#                             video_path=VIDEO_PATH,
#                             frame_dir=FRAME_DIR,
#                             output_path=OUTPUT_PATH,
#                             data_csv_path=DATA_CSV_PATH,
#                             dotsize=DOTSIZE,
#                             colormap=COLORMAP,
#                             bodyparts2plot=BODYPARTS_STR,
#                             trailpoints=TRAILPOINTS,
#                             choose_from_top_or_random=TOP_OR_RANDOM)

# # THEN VISUALIZE FEATURES

# for csv_file in CSVFILE_OF_INTEREST:
#     do_feature_visualization(
#         csvfile=csv_file,
#         csvfolder_path=CSVFOLDER_PATH, 
#         clf_sav_path=CLF_SAV_PATH, 
#         computed_feature_saving_path=COMPUTED_FEATURE_SAVING_PATH, 
#         figure_saving_path=FIGURE_SAVING_PATH, 
#         predictions_path=PREDICTIONS_PATH,
#         bodyparts=BODYPARTS, 
#         groups_to_show=GROUPS_TO_SHOW, 
#         relative_placement_pairs=RELATIVE_PLACEMENT_PAIRS, 
#         relative_angle_pairs=RELATIVE_ANGLE_PAIRS, 
#         displacement_bodyparts=DISPLACEMENT_BODYPARTS, 
#         show_figure=SHOW_FIGURE,
#         save_figure=SAVE_FIGURE,
#         logscale=LOGSCALE, 
#         brute_thresholding=BRUTE_THRESHOLDING
#     )


def main(bsoid_prefix: str,
         scorer_name : str,
         group_name: str,
         csvfolder_path: str,
         video_path: str,
         clf_sav_folder: str,
         predictions_path: str,
         labeled_csv_folder: str,
         saving_folder : str,
         colormap: str="rainbow",
         dotsize: int=7,
         fps: int=40,
         counts: int=5,
         output_fps: int=40,
         trailpoints: int=0,
         groups_to_show: List[int]=[],
         bodyparts: List[Bodypart]=BODYPARTS,
         bodyparts_str: List[str]=BODYPARTS_STR,
         relative_placement_pairs: List[Tuple[Bodypart]]=RELATIVE_PLACEMENT_PAIRS,
         relative_angle_pairs: List[Tuple[Bodypart]]=RELATIVE_ANGLE_PAIRS,
         displacement_bodyparts: List[Bodypart]=DISPLACEMENT_BODYPARTS,
         top_or_random: str="top",
         filtering_noise_max_length: int=5,
         min_desired_bout_length: int=200,
         brute_thresholding: bool=False,
         show_figure: bool=False,
         save_figure: bool=True
         ):
    """
    Quantifies behavior snippets, generates behavior bits, and visualizes features.

    :param str bsoid_prefix: Prefix to the B-SOID network we analyze.
    :param str scorer_name: Name of the scorer of the DLC project.
    :param str group_name: Name of the group shown in figurs of label occurrence quantification.
    :param str csvfolder_path: Path to the folder containing DLC CSV files.
    :param str video_path: Path to the video file used to generate behavior bits.
    :param str clf_sav_folder: Path to where the random forest classifier is stored.
    :param str predictions_path: Path to BSOID predictions file possibly holding already-computed features.
    :param str labeled_csv_folder: Path to the folder holding labeled CSVs by the given network.
    :param str saving_folder: Path to folder to which we save all generated results.
    :param str colormap: Colormap for generating behavior bits (default is "rainbow").
    :param int dotsize: Size of dots for generating behavior bits (default is 7).
    :param int fps: Frames per second at which DLC csvs were recorded. (default is 40).
    :param int counts: How many examples of behavior bits we generate (default is 5).
    :param int output_fps: Output frames per second for behavior bit video (default is 40).
    :param int trailpoints: Number of trail points for behavior bit video (default is 0).
    :param List[int] groups_to_show: Groups of features to show in feature plot (default is every label).
    :param List[Bodypart] bodyparts: List of body parts for feature visualization (default is BODYPARTS).
    :param List[str] bodyparts_str: List of body parts for behavior bits as strings (default is BODYPARTS_STR).
    :param List[Tuple[Bodypart]] relative_placement_pairs: Pairs of body parts for calculating 
    relative placement features in feature visualization (default is RELATIVE_PLACEMENT_PAIRS).
    :param List[Tuple[Bodypart]] relative_angle_pairs: Pairs of body parts for calculating 
    relative angle features in feature visualiation (default is RELATIVE_ANGLE_PAIRS).
    :param List[Bodypart] displacement_bodyparts: Body parts for displacement calculations 
    in feature visualization (default is DISPLACEMENT_BODYPARTS).
    :param str top_or_random: Whether to choose the top behavior bits in length, or random ones, 
    when generating them (default is "top").
    :param int filtering_noise_max_length: Maximum length for noise filtering before generating
    behavior bits (default is 5).
    :param int min_desired_bout_length: Minimum desired bout length when randomly choosing 
    and generating behavior bits (default is 200).
    :param bool brute_thresholding: Whether to use brute thresholding when rendering 
     feature visualization plots (default is False).
    :param bool show_figure: Whether to show generated figures (default is False).
    :param bool save_figure: Whether to save generated figures (default is True).
    """

    # setting up variables and folders
    ###################
    # 1. PATHS
    
    # the path to the network folders generated
    network_name = f'{scorer_name}_{bsoid_prefix.replace("-", "")}'

    # files to compute feature histograms for - held in csvfolder_path
    csvfile_of_interest = [file for file in os.listdir(csvfolder_path) if file.endswith('.csv')]

    # path to classifier
    clf_filename = f"{network_name}_randomforest.sav"
    clf_sav_path = os.path.join(clf_sav_folder, clf_filename)

    # directories holding saved figures / computed features & labels (for later speedup)
    figure_saving_path = os.path.join(saving_folder, RESULT_FOLDER, "figures", network_name)
    computed_feature_saving_path = os.path.join(saving_folder, RESULT_FOLDER, "feats_labels", network_name)

    ######################################
    # GENERATE BEHAVIOR SNIPPETS

    #####DEFINING FILE OF INTEREST#####
    csv_from_video_path = os.path.basename(video_path.replace('.mp4', ''))
    file_of_interest = next((filename for filename in csvfile_of_interest 
                            if csv_from_video_path in filename), None) # choose first csv at random
    if file_of_interest is None:
        raise Exception("Variable VIDEO_PATH has to be set such that it matches at least " + 
                        "one CSV in the folder given by CSVFOLDER_PATH; but failed... \n" + 
                        f"not-found video file: {csv_from_video_path}.")

    #####DEFINING OUTPUT FOLDER#####
    output_folder = os.path.join(saving_folder, RESULT_FOLDER, 'videos', network_name, 'bits')

    #####DEFINING OTHER IMPORTANT FOLDERS#####
    frame_dir = os.path.join(saving_folder, RESULT_FOLDER, 'pngs', get_mousename(file_of_interest))
    data_csv_path = os.path.join(csvfolder_path, file_of_interest)
    labeled_csv_path = os.path.join(labeled_csv_folder, 
                                    [file for file in os.listdir(labeled_csv_folder) 
                                    if file_of_interest in str(file)][0])
    output_path  = os.path.join(output_folder, get_mousename(file_of_interest))

    # GENERATE THE IMPORTANT FOLDERS IF NOT THERE
    toCreate = [frame_dir, output_folder, figure_saving_path, computed_feature_saving_path]
    for tc in toCreate:
        Path(tc).mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Then into execution
    def convert_folder_to_list_of_label(folder: str):
        csvs = [os.path.join(folder, file) for file in os.listdir(folder) if '.csv' in file]
        labels = [read_BSOID_labeled_csv(csv)[0] for csv in csvs]
        return labels

    labels = convert_folder_to_list_of_label(labeled_csv_folder)

    quantify_label_occurrence_and_length_distribution(
        group_of_labels=[labels],
        group_names=[group_name],
        save_dir=figure_saving_path,
        save_name=f"{group_name}.png",
        use_logscale=False,
        save_figure=save_figure,
        show_figure=show_figure
    )

    print("Generating behavior bits...")

    labels = extract_label_from_labeled_csv(labeled_csv_path)

    print("Pre-filtering...")
    filtered = filter_bouts_smaller_than_N_frames(labels, n=filtering_noise_max_length)
    print("Done!")

    create_labeled_behavior_bits(labels=filtered,
                                 crit=min_desired_bout_length / fps,
                                 counts=counts,
                                 output_fps=output_fps,
                                 video_path=video_path,
                                 frame_dir=frame_dir,
                                 output_path=output_path,
                                 data_csv_path=data_csv_path,
                                 dotsize=dotsize,
                                 colormap=colormap,
                                 bodyparts2plot=bodyparts_str,
                                 trailpoints=trailpoints,
                                 choose_from_top_or_random=top_or_random)

    for csv_file in csvfile_of_interest:
        do_feature_visualization(
            csvfile=csv_file,
            csvfolder_path=csvfolder_path,
            clf_sav_path=clf_sav_path,
            computed_feature_saving_path=computed_feature_saving_path,
            figure_saving_path=figure_saving_path,
            predictions_path=predictions_path,
            bodyparts=bodyparts,
            groups_to_show=groups_to_show,
            relative_placement_pairs=relative_placement_pairs,
            relative_angle_pairs=relative_angle_pairs,
            displacement_bodyparts=displacement_bodyparts,
            show_figure=show_figure,
            save_figure=save_figure,
            logscale=True,
            brute_thresholding=brute_thresholding
        )

if __name__ == "__main__":
    # name of network we analyze
    BSOID_PREFIX = 'Apr-18-2024V2'
    # name of the scorer of the project
    SCORER_NAME = "Akira"
    # path to which we save all generated results in organized ways
    SAVING_FOLDER = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID"

    # path to DLC csv files
    CSVFOLDER_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr182024V2\CSV files\filt"

    # path to where the random forest classifier is stored
    CLF_SAV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr182024V2\output"
    # path to a predictions file that possibly holds already-computed features
    # might be deprecated if we implement an auto search within CSVFOLDER_PATH
    PREDICTIONS_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr182024V2\output\Apr-18-2024V2_predictions.sav"

    # BEHAVIOR BIT GENERATION:
    # path to folder holding labeled csvs by the given network
    LABELED_CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\Q175\Apr182024V2\CSV files\filt\BSOID"
    # path to video we use to generate behavior bits - has to be matching at least
    # one of the csv files contained under CSVFOLDER_PATH
    VIDEO_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Q175 Male Open Field Behaviour\Switched\WT\20220228223808_320151_m1_openfield.mp4"

    # LABEL AND BEHAVIOR BIT OCCURRENCE PLOT:
    GROUP_NAME = "Q175" # name of group we show in the figures

    main(bsoid_prefix=BSOID_PREFIX,
         scorer_name=SCORER_NAME,
         group_name=GROUP_NAME,
         csvfolder_path=CSVFOLDER_PATH,
         video_path=VIDEO_PATH,
         clf_sav_folder=CLF_SAV_FOLDER,
         predictions_path=PREDICTIONS_PATH,
         labeled_csv_folder=LABELED_CSV_FOLDER, 
         saving_folder=SAVING_FOLDER)
    