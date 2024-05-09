# Author: Akira Kudo
# Created: 2024/04/10
# Last Updated: 2024/05/09

# A file that experiments around with filtering based on impossible speed

import os

from dlc_io.utils import read_dlc_csv_file
from do_make_video_from_dlc_and_png import extract_frames_and_construct_video_from_csv, extract_frames_and_construct_video_from_dataframe, extract_frames_and_construct_video_from_two_dataframes
from temper_with_csv_and_hdf.data_filtering.filter_based_on_boolean_array import filter_based_on_boolean_array
from temper_with_csv_and_hdf.data_filtering.identify_paw_noise import identify_bodypart_noise_in_rest, identify_bodypart_noise_by_impossible_speed
from temper_with_csv_and_hdf.data_filtering.remove_dlc_entry_by_probability import remove_dlc_entry_by_probability_from_dataframe
from utils_to_be_replaced_oneday import bodypart_abbreviation_dict, get_mousename


# VIDEO_PATH = r"C:\Users\mashi\Desktop\temp\YAC\videos\20230113142714_392607_m1_openfield.mp4"
VIDEO_PATH = r"C:\Users\mashi\Desktop\temp\Q175\videos\20220228223808_320151_m1_openfield.mp4"

# CSV_FOLDER = r"C:\Users\mashi\Desktop\temp\YAC\preBSOID_csv\snapshot-2010000\unfiltered"
CSV_FOLDER = r"C:\Users\mashi\Desktop\temp\Q175\csvs"

# CSV_FILENAME = "20230113142714_392607_m1_openfieldDLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_2100000.csv"
CSV_FILENAME = "20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_2060000.csv"

CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILENAME)

# IMG_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos\extracted"
IMG_DIR = r"C:\Users\mashi\Desktop\temp\Q175\videos\extracted"

# OUTPUT_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\videos\generated"
OUTPUT_DIR = r"C:\Users\mashi\Desktop\temp\Q175\videos\generated"

FILTER_LIKELIHOOD = 0.9

START, END = 0, 1500
FPS = 20
IMPOSSIBLE_SPEED_THRESHOLD = 60
BODYPARTS = [
    'snout',
    'rightforepaw',
    'leftforepaw',
    'righthindpaw', 
    'lefthindpaw', 
    'tailbase'
    ]

FILTER_MODE = "linear"

mousename = get_mousename(VIDEO_PATH)

# first extract non-filtered sequence
non_filtered_video_name = f"{mousename}_{START}to{END}_{FPS}fps.mp4"
if not os.path.exists(os.path.join(OUTPUT_DIR, non_filtered_video_name)):
    print("First non-filtered:")
    extract_frames_and_construct_video_from_csv(
        video_path=VIDEO_PATH, 
        csv_path=CSV_PATH, 
        img_dir=IMG_DIR, output_dir=OUTPUT_DIR, 
        start=START, end=END, fps=FPS,
        output_name=non_filtered_video_name
    )
# also filtered based on likelihood
likelihood_filtered_video_name = f"{mousename}_{START}to{END}_{FPS}fps_filterByLikelihood_{FILTER_LIKELIHOOD}.mp4"
if not os.path.exists(os.path.join(OUTPUT_DIR, likelihood_filtered_video_name)):
    print(f"Also filtered based on likelihood {FILTER_LIKELIHOOD}:")
    df = read_dlc_csv_file(dlc_path=CSV_PATH)
    likelihood_filtered_df = remove_dlc_entry_by_probability_from_dataframe(
        df, threshold=FILTER_LIKELIHOOD, bodyparts=['snout']
        )
    extract_frames_and_construct_video_from_dataframe(
        video_path=VIDEO_PATH, 
        dlc_df=likelihood_filtered_df,
        img_dir=IMG_DIR, output_dir=OUTPUT_DIR, 
        start=START, end=END, fps=FPS,
        output_name=likelihood_filtered_video_name
    )

# then extract filtered sequence
df = read_dlc_csv_file(dlc_path=CSV_PATH)
filtered_df = df
# first based on noises during rest
"""
for bpt in BODYPARTS:
    _, wrong_frames = identify_bodypart_noise_in_rest(
        dlc_csv_path=CSV_PATH,
        bodypart=bpt,
        show_start=START,
        show_end=END,
        # threshold=0,
        )
    truncated_wrng_frms = wrong_frames[START:END+1]
    filtered_df = filter_based_on_boolean_array(
        bool_arr=truncated_wrng_frms, df=filtered_df, bodyparts=[bpt], filter_mode="latest",
        start=START, end=END
    )
"""

# then observing impossible speeds
impossible_move_df = identify_bodypart_noise_by_impossible_speed(
    dlc_csv_path=CSV_PATH, bodyparts=BODYPARTS, start=START, end=END,
    # threshold=IMPOSSIBLE_SPEED_THRESHOLD
    )

for bpt in BODYPARTS:
    bpt_bool_arr = impossible_move_df[(bpt, "loc_wrng")].to_numpy()

    old_df = filtered_df
    filtered_df = filter_based_on_boolean_array(
        bool_arr=bpt_bool_arr, df=filtered_df, bodyparts=[bpt], filter_mode=FILTER_MODE,
        start=START, end=END
    )

print("Then filtered:")

abbreviations = [bodypart_abbreviation_dict[bpt] for bpt in BODYPARTS]
extract_frames_and_construct_video_from_two_dataframes(
    video_path=VIDEO_PATH, 
    dlc_df1=filtered_df, 
    dlc_df2=df,
    img_dir=IMG_DIR, output_dir=OUTPUT_DIR, 
    start=START, end=END, fps=FPS,
    )

# extract_frames_and_construct_video_from_dataframe(
#     video_path=VIDEO_PATH, 
#     dlc_df=filtered_df, 
#     img_dir=IMG_DIR, output_dir=OUTPUT_DIR, 
#     start=START, end=END, fps=FPS,
#     output_name=f"{mousename}_{START}to{END}_{FPS}fps_{'_'.join(abbreviations)}_filtered_speed{IMPOSSIBLE_SPEED_THRESHOLD}.mp4"
# )