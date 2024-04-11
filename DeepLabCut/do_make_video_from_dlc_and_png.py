# Author: Akira Kudo
# Created: 2024/04/08
# Last Updated: 2024/04/10

import os

from temper_with_csv_and_hdf.data_filtering.make_video_from_dlc_and_images import make_video_from_dlc_and_images
from video_related.extract_image_from_mp4 import extract_frame_by_number
from utils_to_be_replaced_oneday import get_mousename

VIDEO_PATH = r"C:\Users\mashi\Desktop\temp\YAC\videos\20230113142714_392607_m1_openfield.mp4"
video_mousename = get_mousename(VIDEO_PATH)

CSV_FOLDER = r"C:\Users\mashi\Desktop\temp\YAC\preBSOID_csv"
# r"C:\Users\mashi\Desktop\RaymondLab\Experiments\B-SOID\Q175_Network\Q175_csv"
#r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\B-SOID\Q175 Open Field CSVs\WT\snapshot950000"
CSV_FILENAME = "20230113142714_392607_m1_openfieldDLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000_filtered.csv"
# r"20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"
#r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_950000.csv"
csv_mousename = get_mousename(CSV_FILENAME)

if video_mousename != csv_mousename:
    raise Exception(f"Mouse names {video_mousename} and {csv_mousename} don't match...")
else:
    MOUSENAME = csv_mousename

CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILENAME)

IMG_DIR = r"C:\Users\mashi\Desktop\temp\YAC\videos\extracted\{}".format(MOUSENAME)
IMG_NAME = f"{MOUSENAME}_.jpg"

START, END = 200, 400
FPS = 5

OUTPUT_DIR = r"C:\Users\mashi\Desktop\temp\YAC\videos\generated"
OUTPUT_NAME = f"{MOUSENAME}_{START}to{END}_{FPS}fps.mp4"


if not os.path.exists(IMG_DIR):
    os.mkdir(IMG_DIR)

try:
    print("Generating the video...")
    make_video_from_dlc_and_images(dlc_path=CSV_PATH,
                                    img_dir=IMG_DIR,
                                    frame_start=START, frame_end=END,
                                    fps=FPS,
                                    output_dir=OUTPUT_DIR, output_name=OUTPUT_NAME)
except Exception as e:
    # not the most robust, so please don't run this code in crazy ways
    if str(e).startswith("An image with extension "):
        print("Extracting frames.")
        extract_frame_by_number(input_file=VIDEO_PATH, 
                                output_file=os.path.join(IMG_DIR, IMG_NAME), 
                                frame_numbers=list(range(START, END+1)))
        print("Done!")
        print("-----------------------------------")
        print("Generating the video...")
        make_video_from_dlc_and_images(dlc_path=CSV_PATH,
                                        img_dir=IMG_DIR,
                                        frame_start=START, frame_end=END,
                                        fps=FPS,
                                        output_dir=OUTPUT_DIR, output_name=OUTPUT_NAME)
print("Done!")