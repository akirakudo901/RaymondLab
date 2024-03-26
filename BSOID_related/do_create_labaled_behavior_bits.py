# Author: Akira Kudo
# Created: 2024/03/21
# Last Updated: 2024/03/21

import os

from label_behavior_bits.create_labeled_behavior_bits import create_labeled_behavior_bits, extract_label_from_labeled_csv, filter_bouts_smaller_than_N_frames

FPS = 40
FILTERING_NOISE_MAX_LENGTH = 5 # max length of noise filtered via filter_bouts_smaller_than_N_frames
MIN_DESIRED_BOUT_LENGTH = 500
COUNTS = 5
OUTPUT_FPS = 40
TRAILPOINTS = 0
TOP_OR_RANDOM = "top" #"random"

DOTSIZE = 7
COLORMAP = "rainbow" # obtained from config.yaml on DLC side
# we exclude "belly" as it isn't used to classify in this B-SOID
BODYPARTS = ["snout",        "rightforepaw", "leftforepaw", 
                "righthindpaw", "lefthindpaw",  "tailbase"] 

#####DEFINING FILE OF INTEREST#####
FILE_OF_INTEREST = r"312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
# r"20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
# r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"

LABELED_PREFIX = r"Mar-10-2023labels_pose_40Hz"     # r"Feb-27-2024labels_pose_40Hz"

#####DEFINING OUTPUT FOLDER#####
OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\previous\B-SOID STUFF\BoutVideoBits\YAC128\labeled_five_length_bout_filtered"
# r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\previous\B-SOID STUFF\BoutVideoBits\labeled"

#####DEFINING OTHER IMPORTANT FOLDERS#####
MOUSETYPE_FOLDER = r"D:\B-SOID\Leland B-SOID YAC128 Analysis" + \
                   r"\YAC128\YAC128" # r"\Q175\WT"

FRAME_DIR = os.path.join(MOUSETYPE_FOLDER,  
                         "pngs", # r"csv\pngs",
                         FILE_OF_INTEREST.replace(".csv", ""))

DATA_CSV_PATH = os.path.join(MOUSETYPE_FOLDER,
                             r"CSV files", # r"csv"
                             FILE_OF_INTEREST)

LABELED_CSV_PATH = os.path.join(MOUSETYPE_FOLDER,  
                                "CSV files", "BSOID", #r"csv/BSOID/Feb-27-2024"
                                LABELED_PREFIX + FILE_OF_INTEREST)

OUTPUT_PATH   = os.path.join(OUTPUT_FOLDER, FILE_OF_INTEREST.replace(".csv", ""))


os.mkdir(OUTPUT_FOLDER)

labels = extract_label_from_labeled_csv(LABELED_CSV_PATH)

print("Pre-filtering...")
filtered = filter_bouts_smaller_than_N_frames(labels, n=FILTERING_NOISE_MAX_LENGTH)
print("Done!")

create_labeled_behavior_bits(labels=filtered, 
                             crit=MIN_DESIRED_BOUT_LENGTH / FPS, 
                             counts=COUNTS,
                             output_fps=OUTPUT_FPS, 
                             frame_dir=FRAME_DIR,
                             output_path=OUTPUT_PATH,
                             data_csv_path=DATA_CSV_PATH,
                             dotsize=DOTSIZE,
                             colormap=COLORMAP,
                             bodyparts2plot=BODYPARTS,
                             trailpoints=TRAILPOINTS,
                             choose_from_top_or_random=TOP_OR_RANDOM)