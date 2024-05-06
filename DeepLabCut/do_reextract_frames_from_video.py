# Author: Akira Kudo
# Created: 2024/04/16
# Last Updated: 2024/04/25

import os

from video_related.reextract_frames_from_video import reextract_frames_from_video

ABOVE_IMAGE_DIR = r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown Halfscale-Akira-2024-03-15/labeled-data"
IMG_DIR_PATH = [
    "20220211070325_301533_f3__rescaled_by_0point5",
    "20220228223808_320151_m1_openfield_rescaled_by_0point5",
    "20220228231804_320151_m2_openfield_rescaled_by_0point5",
    "20230102092905_363451_f1_openfield_rescaled_by_0point5",
    "20230107123308_362816_m1_openfield_rescaled_by_0point5",
    "20230107131118_363453_m1_openfield_rescaled_by_0point5"
]

ABOVE_VIDEO_PATH = r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown Halfscale-Akira-2024-03-15/videos"
VIDEO_PATH = [f'{video}.mp4' for video in IMG_DIR_PATH]

ABOVE_OUTPATH_DIR = r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown Halfscale-Akira-2024-03-15/labeled-data/reextracted_frames"
OUTDIR_PATH = IMG_DIR_PATH

for imgdir, videopath, outdir in zip(IMG_DIR_PATH, VIDEO_PATH, OUTDIR_PATH):
    reextract_frames_from_video(img_dir_path=os.path.join(ABOVE_IMAGE_DIR, imgdir), 
                                video_path=os.path.join(ABOVE_VIDEO_PATH, videopath), 
                                outdir_path=os.path.join(ABOVE_OUTPATH_DIR, outdir))