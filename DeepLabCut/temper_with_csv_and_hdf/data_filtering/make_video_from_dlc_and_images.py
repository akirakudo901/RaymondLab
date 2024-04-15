# Author: Akira Kudo
# Created: 2024/04/08
# Last Updated: 2024/04/10

import os
import re
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.draw import disk

from dlc_io.utils import read_dlc_csv_file

IMG_EXTENSION = '.jpg'
COLORMAP = "rainbow"
DOTSIZE = 7
NUM_TRAILPOINTS = 0

def make_video_from_dlc_csv_and_images(
        dlc_path : str, img_dir : str,
        frame_start : int, frame_end : int,
        fps : int,
        output_dir : str, output_name : str):
    """
    Creates a video from the given dlc & pngs found under given 
    directories, as well as the specified starting & ending frames.
    Video is created in specified frame-per-second.

    :param str dlc_path: Path to DLC holding bodypart data.
    :param str img_dir: Path to directory holding images, each named 
    so that they contain at their end one number specifying which frame 
    in the video it is. E.g. 'image230_12.png' is numbered 12.
    :param int frame_start: The start of extracted frame sequences, inclusive.
    :param int frame_end: The end of extracted frame sequences, inclusive.
    :param int fps: The frame-per-second for the produced video.
    :param str output_dir: The path to the output directory.
    :param str output_name: The name of the outputted file.
    """
    df = read_dlc_csv_file(dlc_path=dlc_path, include_scorer=False)
    make_video_from_dlc_dataframe_and_images(
        df=df, img_dir=img_dir, frame_start=frame_start, 
        frame_end=frame_end, fps=fps, output_dir=output_dir, 
        output_name=output_name
        )

def make_video_from_dlc_dataframe_and_images(
        df : pd.DataFrame, img_dir : str,
        frame_start : int, frame_end : int,
        fps : int,
        output_dir : str, output_name : str
        ):
    """
    Creates a video from the given dataframe as extracted from a dlc csv, 
    pngs found under given directories, as well as the specified 
    starting & ending frames. Video is created in specified frame-per-second.

    :param pd.DataFrame df: pandas DataFrame holding bodypart data.
    :param str img_dir: Path to directory holding images, each named 
    so that they contain at their end one number specifying which frame 
    in the video it is. E.g. 'image230_12.png' is numbered 12.
    :param int frame_start: The start of extracted frame sequences, inclusive.
    :param int frame_end: The end of extracted frame sequences, inclusive.
    :param int fps: The frame-per-second for the produced video.
    :param str output_dir: The path to the output directory.
    :param str output_name: The name of the outputted file.
    """
    df_x, df_y, _ = df.values.reshape((len(df), -1, 3)).T
    
    # first check that corresponding images already exist in img_dir
    frame_indices_in_pngdir = [
        (int(re.findall(r'\d+', filename)[-1]), filename) # frame_num : filename
        for filename in os.listdir(img_dir)
        if filename.endswith(IMG_EXTENSION)
        ]
    # remove any non-match (None results for first entry)
    frame_indices_in_pngdir = [f for f in frame_indices_in_pngdir if f[0] is not None]
    # make a dictionary out of (filename : frame_num) pairs
    frame_number_to_filename = dict(frame_indices_in_pngdir)
    for img_idx in range(frame_start, frame_end + 1):
        if img_idx not in frame_number_to_filename.keys():
            raise Exception(f"An image with extension {IMG_EXTENSION} and " +
                            f"index {img_idx} didn't exist in img_dir; " + 
                             "please generate such image first!")
    
    # then create the video using cv2
    # first get the color
    all_bpts = np.array(range(7)) #all body parts; 7 of them
    colorclass = plt.cm.ScalarMappable(cmap=COLORMAP)
    C = colorclass.to_rgba(np.linspace(0, 1, len(all_bpts)))
    colors = (C[:, :3] * 255).astype(np.uint8)
    # map bodyparts to color
    bpts2color = [(ind, colors[ind]) for ind in all_bpts]
    
    # ready the video creator
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(img_dir, 
                                    frame_number_to_filename[frame_start]))
    height, width, _ = frame.shape
    
    # generate the video
    try:
        video = cv2.VideoWriter(os.path.join(output_dir, output_name), fourcc, fps, (width, height))
        for img_idx in range(frame_start, frame_end + 1):
            imgfile = frame_number_to_filename[img_idx]
            read_img = cv2.imread(os.path.join(img_dir, imgfile))
            labeled_img = label_bodyparts_on_single_frame(read_img,
                                                          index=img_idx,
                                                          x_filtered=df_x,
                                                          y_filtered=df_y,
                                                          dotsize=DOTSIZE,
                                                          bpts2color=bpts2color,
                                                          trailpoints=NUM_TRAILPOINTS)
            video.write(labeled_img)
        cv2.destroyAllWindows()
        video.release()
    except:
        traceback.print_exc() 
        print(f"Generation failed...")

def label_bodyparts_on_single_frame(
    image : np.ndarray,
    index : int,
    x_filtered : np.ndarray,
    y_filtered : np.ndarray,
    dotsize,
    bpts2color,
    trailpoints : int = 0
):
    """
    Creating individual frames with labeled body parts.
    image: np.ndarray
        The image we label with bodyparts.
    
    index: int
        The index of the image frame within the video.

    bpts2color: list of tuple, (integer, color)
        This specifies a list of tuples, where each tuple is a set of an integer
        indicating the column number for the body part in question, as well as 
        its corresponding color.

    trailpoints: int
        Number of previous frames whose body parts are plotted in a frame (for displaying 
        history). Default is set to 0.

    """
    # this function USED TO ADJUST for the fact that:
    # 1 - frames are reduced in dimension by twice when being extracted from the video
    #     and stored into saving folders by B-SOID
    # NOW IT IS NOT, AS WE ARE RENDERING NORMAL-RATIO VIDEOS
    def draw_size_adjusted_disk(image, color, center_x, center_y, dotsize, ny, nx):
        ratio = 1
        rr, cc = disk((center_y*ratio, center_x*ratio), 
                      dotsize*ratio, 
                      shape=(ny, nx))
        image[rr, cc] = color

    ny, nx = image.shape[1], image.shape[0]
    
    with np.errstate(invalid="ignore"):
        for ind, color in bpts2color:
            if trailpoints < 0: trailpoints = 0
            for k in range(trailpoints, -1, -1):
                rendered_idx = index - k
                if rendered_idx >= 0:
                    draw_size_adjusted_disk(image, color, 
                        x_filtered[ind, rendered_idx], y_filtered[ind, rendered_idx], 
                        dotsize, ny, nx)
    
    return image