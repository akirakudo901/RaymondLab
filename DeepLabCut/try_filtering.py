# Author: Akira Kudo
# Created: 2024/04/10
# Last Updated: 2024/04/11

# A file that experiments around with filtering based on impossible speed

import os

from dlc_io.utils import read_dlc_csv_file
from do_make_video_from_dlc_and_png import extract_frames_and_construct_video_from_csv, extract_frames_and_construct_video_from_dataframe
from temper_with_csv_and_hdf.data_filtering.filter_based_on_boolean_array import filter_based_on_boolean_array
from temper_with_csv_and_hdf.data_filtering.identify_paw_noise import identify_bodypart_noise_in_rest
from utils_to_be_replaced_oneday import get_mousename

VIDEO_PATH = r"C:\Users\mashi\Desktop\temp\YAC\videos\20230113142714_392607_m1_openfield.mp4"

CSV_FOLDER = r"C:\Users\mashi\Desktop\temp\YAC\preBSOID_csv"
# r"C:\Users\mashi\Desktop\RaymondLab\Experiments\B-SOID\Q175_Network\Q175_csv"
#r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\B-SOID\Q175 Open Field CSVs\WT\snapshot950000"
CSV_FILENAME = "20230113142714_392607_m1_openfieldDLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000_filtered.csv"
# r"20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"
#r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_950000.csv"
CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILENAME)

IMG_DIR = r"C:\Users\mashi\Desktop\temp\YAC\videos\extracted"
OUTPUT_DIR = r"C:\Users\mashi\Desktop\temp\YAC\videos\generated"

START, END = 400, 500
FPS = 10
BODYPARTS = ['rightforepaw', 'leftforepaw']
BODYPARTS_ABBREV = {
    'snout' : 'sn', 'rightforepaw' : 'rfp', 'leftforepaw' : 'lfp',
    'righthindpaw' : 'rhp', 'lefthindpaw' : 'lhp', 'tailbase' : 'tb',
    'belly' : 'bl'
    }

mousename = get_mousename(VIDEO_PATH)

# first extract non-filtered sequence
print("First non-filtered:")
extract_frames_and_construct_video_from_csv(
    video_path=VIDEO_PATH, 
    csv_path=CSV_PATH, 
    img_dir=IMG_DIR, output_dir=OUTPUT_DIR, 
    start=START, end=END, fps=FPS,
    output_name=f"{mousename}_{START}to{END}_{FPS}fps.mp4"
)

# then extract filtered sequence

df = read_dlc_csv_file(dlc_path=CSV_PATH)
filtered_df = df

for bpt in BODYPARTS:
    _, wrong_frames = identify_bodypart_noise_in_rest(
        dlc_csv_path=CSV_PATH,
        bodypart=bpt,
        show_start=START,
        show_end=END,
        # threshold=0,
        )
    filtered_df = filter_based_on_boolean_array(
        bool_arr=wrong_frames, df=filtered_df, bodyparts=[bpt], filter_mode="latest"
    )

print("Then filtered:")

abbreviations = [BODYPARTS_ABBREV[bpt] for bpt in BODYPARTS]
extract_frames_and_construct_video_from_dataframe(
    video_path=VIDEO_PATH, 
    dlc_df=filtered_df, 
    img_dir=IMG_DIR, output_dir=OUTPUT_DIR, 
    start=START, end=END, fps=FPS,
    output_name=f"{mousename}_{START}to{END}_{FPS}fps_{'_'.join(abbreviations)}_filtered.mp4"
)