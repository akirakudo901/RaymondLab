# Author: Akira Kudo
# Created: 2024/04/08
# Last Updated: 2024/04/22

import os

import pandas as pd

from dlc_io.utils import read_dlc_csv_file
from temper_with_csv_and_hdf.data_filtering.make_video_from_dlc_and_images import make_video_from_dlc_dataframe_and_images, make_video_from_two_dlc_dataframes_and_images
from utils_to_be_replaced_oneday import get_mousename
from video_related.extract_image_from_mp4 import extract_frame_by_number

def extract_frames_and_construct_video_from_csv(
        video_path : str,
        csv_path : str,
        img_dir : str,
        output_dir : str,
        start : int,
        end : int,
        fps : int,
        img_name : str=None,
        output_name : str=None
        ):
    """
    Given relevant paths, extracts frames in the specified
    range from a video, overlays bodyparts data from the
    given DLC data and generates a video.

    :param str video_path: The path to the video we extract frames from.
    :param str csv_path: The csv containing DLC body parts data.
    :param str img_dir: The directory to which we extract frames - a 
    subfolder is created with this mouse's name, to which frames are saved.
    :param str output_dir: The directory the generated video is output.
    :param int start: Start of range of frames extracted, inclusive.
    :param int end: End of range of frames extracted, inclusive.
    :param int fps: Frame-per-second for generated video.
    :param str img_name: Name of extracted frames to be optionally 
    specified; defaults to "{MOUSENAME}_{FRAME_NUMBER}.jpg" if None.
    :param str output_name: Name of the generated video, defaults if None
    to: "{MOUSENAME}_{start}to{end}_{fps}fps.mp4"
    """
    # first get the mouse names, ensuring they match
    video_mousename, csv_mousename = get_mousename(video_path), get_mousename(csv_path)
    if video_mousename != csv_mousename:
        raise Exception(f"Mouse names {video_mousename} and {csv_mousename} don't match...")
    
    df = read_dlc_csv_file(csv_path)
    extract_frames_and_construct_video_from_dataframe(
        video_path=video_path, dlc_df=df, img_dir=img_dir, 
        output_dir=output_dir, start=start, end=end, fps=fps,
        img_name=img_name, output_name=output_name
    )
    
def extract_frames_and_construct_video_from_dataframe(
        video_path : str,
        dlc_df : pd.DataFrame,
        img_dir : str,
        output_dir : str,
        start : int,
        end : int,
        fps : int,
        img_name : str=None,
        output_name : str=None
        ):
    """
    Given relevant paths, extracts frames in the specified
    range from a video, overlays bodyparts data from the
    given DLC dataframe and generates a video.

    :param str video_path: The path to the video we extract frames from.
    :param pd.DataFrame dlc_df: The dataframe containing DLC body parts data.
    :param str img_dir: The directory to which we extract frames - a 
    subfolder is created with this mouse's name, to which frames are saved.
    :param str output_dir: The directory the generated video is output.
    :param int start: Start of range of frames extracted, inclusive.
    :param int end: End of range of frames extracted, inclusive.
    :param int fps: Frame-per-second for generated video.
    :param str img_name: Name of extracted frames to be optionally 
    specified; defaults to "{MOUSENAME}_{FRAME_NUMBER}.jpg" if None.
    :param str output_name: Name of the generated video, defaults if None
    to: "{MOUSENAME}_{start}to{end}_{fps}fps.mp4"
    """
    # first get the mouse name
    mousename = get_mousename(video_path)
    # then set default arguments
    if img_name is None:
        img_name = f"{mousename}_.jpg"
    if output_name is None:
        output_name = f"{mousename}_{start}to{end}_{fps}fps.mp4"

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    try:
        print("Generating the video...")
        make_video_from_dlc_dataframe_and_images(
            df=dlc_df, img_dir=img_dir, 
            frame_start=start, frame_end=end,
            fps=fps, output_dir=output_dir, output_name=output_name
            )
    except Exception as e:
        # not the most robust, so please don't run this code in crazy ways
        if str(e).startswith("An image with extension "):
            print("Extracting frames.")
            extract_frame_by_number(input_file=video_path, 
                                    output_file=os.path.join(img_dir, img_name), 
                                    frame_numbers=list(range(start, end+1)))
            print("Done!")
            print("-----------------------------------")
            print("Generating the video...")
            make_video_from_dlc_dataframe_and_images(
                df=dlc_df, img_dir=img_dir, 
                frame_start=start, frame_end=end,
                fps=fps, output_dir=output_dir, output_name=output_name
                )
    print("Done!")

def extract_frames_and_construct_video_from_two_csvs(
        video_path : str,
        csv1_path : str,
        csv2_path : str,
        img_dir : str,
        output_dir : str,
        start : int,
        end : int,
        fps : int,
        img_name : str=None,
        output_name : str=None
        ):
    """
    Given relevant paths, extracts frames in the specified
    range from a video, overlays bodyparts data from two DLC 
    data given, and generates a video.

    :param str video_path: The path to the video we extract frames from.
    :param str csv1_path: One csv of DLC body parts data - rendered as circle.
    :param str csv2_path: Other csv of DLC body parts data - rendered as triangle.
    :param str img_dir: The directory to which we extract frames - a 
    subfolder is created with this mouse's name, to which frames are saved.
    :param str output_dir: The directory the generated video is output.
    :param int start: Start of range of frames extracted, inclusive.
    :param int end: End of range of frames extracted, inclusive.
    :param int fps: Frame-per-second for generated video.
    :param str img_name: Name of extracted frames to be optionally 
    specified; defaults to "{MOUSENAME}_{FRAME_NUMBER}.jpg" if None.
    :param str output_name: Name of the generated video, defaults if None
    to: "{MOUSENAME}_{start}to{end}_{fps}fps.mp4"
    """
    # first get the mouse names, ensuring they match
    video_mousename = get_mousename(video_path)
    csv1_mousename, csv2_mousename = get_mousename(csv1_path), get_mousename(csv2_path)
    if video_mousename != csv1_mousename:
        raise Exception(f"Video mouse name {video_mousename} and csv mouse name {csv1_mousename} don't match...")
    elif video_mousename != csv2_mousename:
        raise Exception(f"Video mouse name {video_mousename} and csv2 mouse name {csv2_mousename} don't match...")
    elif csv1_mousename != csv2_mousename:
        raise Exception(f"CSV mouse names {csv1_mousename} and {csv2_mousename} don't match...")
    
    df1, df2 = read_dlc_csv_file(csv1_mousename), read_dlc_csv_file(csv2_mousename)
    extract_frames_and_construct_video_from_two_dataframes(
        video_path=video_path, dlc_df1=df1, dlc_df2=df2, img_dir=img_dir, 
        output_dir=output_dir, start=start, end=end, fps=fps,
        img_name=img_name, output_name=output_name
    )
    
def extract_frames_and_construct_video_from_two_dataframes(
        video_path : str,
        dlc_df1 : pd.DataFrame,
        dlc_df2 : pd.DataFrame,
        img_dir : str,
        output_dir : str,
        start : int,
        end : int,
        fps : int,
        img_name : str=None,
        output_name : str=None
        ):
    """
    Given relevant paths, extracts frames in the specified
    range from a video, overlays bodyparts data from two
    given DLC dataframes and generates a video.

    :param str video_path: The path to the video we extract frames from.
    :param pd.DataFrame dlc_df1: Dataframe containing DLC data - rendered as circle.
    :param pd.DataFrame dlc_df2: Dataframe containing DLC data - rendered as triangle.
    :param str img_dir: The directory to which we extract frames - a 
    subfolder is created with this mouse's name, to which frames are saved.
    :param str output_dir: The directory the generated video is output.
    :param int start: Start of range of frames extracted, inclusive.
    :param int end: End of range of frames extracted, inclusive.
    :param int fps: Frame-per-second for generated video.
    :param str img_name: Name of extracted frames to be optionally 
    specified; defaults to "{MOUSENAME}_{FRAME_NUMBER}.jpg" if None.
    :param str output_name: Name of the generated video, defaults if None
    to: "{MOUSENAME}_{start}to{end}_{fps}fps.mp4"
    """
    # first get the mouse name
    mousename = get_mousename(video_path)
    # then set default arguments
    if img_name is None:
        img_name = f"{mousename}_.jpg"
    if output_name is None:
        output_name = f"{mousename}_{start}to{end}_{fps}fps.mp4"

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    try:
        print("Generating the video...")
        make_video_from_two_dlc_dataframes_and_images(
            df1=dlc_df1, df2=dlc_df2, img_dir=img_dir, 
            frame_start=start, frame_end=end,
            fps=fps, output_dir=output_dir, output_name=output_name
        )
    except Exception as e:
        # not the most robust, so please don't run this code in crazy ways
        if str(e).startswith("An image with extension "):
            print("Extracting frames.")
            extract_frame_by_number(input_file=video_path, 
                                    output_file=os.path.join(img_dir, img_name), 
                                    frame_numbers=list(range(start, end+1)))
            print("Done!")
            print("-----------------------------------")
            print("Generating the video...")
            make_video_from_two_dlc_dataframes_and_images(
                df1=dlc_df1, df2=dlc_df2, img_dir=img_dir, 
                frame_start=start, frame_end=end,
                fps=fps, output_dir=output_dir, output_name=output_name
            )
    print("Done!")


if __name__ == "__main__":
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

    START, END = 1000, 1200
    FPS = 10

    extract_frames_and_construct_video_from_csv(
        video_path=VIDEO_PATH, 
        csv_path=CSV_PATH, 
        img_dir=IMG_DIR, output_dir=OUTPUT_DIR, 
        start=START, end=END, fps=FPS
    )
