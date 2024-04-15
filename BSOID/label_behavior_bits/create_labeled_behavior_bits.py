# Author: Akira Kudo
# Render labels on top of identified behavior bit videos.
# THESE CODES WERE ALMOST ENTIRELY TAKEN FROM: 
# https://github.com/DeepLabCut/DeepLabCut/blob/v2.2.0.6/deeplabcut/utils/make_labeled_video.py#L523
# AND:
# https://github.com/YttriLab/B-SOID/blob/master/bsoid_app/bsoid_utilities/videoprocessing.py#L15

import heapq
import os
import random
import re
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.draw import disk
from tqdm import tqdm

from .preprocessing import adp_filt_bsoid_style, extract_label_from_labeled_csv, filter_bouts_smaller_than_N_frames

# Given an image and corresponding coordinates for bodyparts, renders 
# the bodyparts onto the image, to be returned.
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
    # this function adjusts for the fact that:
    # 1 - frames are reduced in dimension by twice when being extracted from the video
    #     and stored into saving folders by B-SOID
    def draw_size_adjusted_disk(image, color, center_x, center_y, dotsize, ny, nx):
        ratio = 1/2
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



# Create a labeled video based on a set of frames
def create_labeled_behavior_bits(labels, 
                                 crit, 
                                 counts,
                                 output_fps, 
                                 frame_dir, 
                                 output_path, 
                                 data_csv_path,
                                 dotsize,
                                 colormap,
                                 bodyparts2plot,
                                 trailpoints,
                                 choose_from_top_or_random : str="random"):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors (~300ms is advised)
    :param counts: scalar, number of generated examples (~5 is advised)
    :param output_fps: integer, frame per second for the output video
    :param frame_dir: string, directory to where you extracted vid images
    :param output_path: string, directory to where you want to store short video examples

    :param choose_from_top_or_random: Whether to choose 'counts' groups at random or from the 
    top N 'counts' in terms of length. If not "random", chooses the top 'counts'.
    """
    print(f"Generating video snippets for: |{os.path.basename(data_csv_path)}| under {data_csv_path}.")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    subfolder = os.path.join(output_path, "random_n" if choose_from_top_or_random == "random" else "top_n")
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)
    # generate limb position labels exactly as filtered by B-SOID (removes low probability predictions)
    Dataframe = pd.read_csv(data_csv_path, index_col=0, header=[0,1,2], skip_blank_lines=False, low_memory=False)
    df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T
    df_x_filt, df_y_filt = adp_filt_bsoid_style(df_x.T, df_y.T, df_likelihood.T, False)
    df_x_filt, df_y_filt = df_x_filt.T, df_y_filt.T
    # get body parts
    bpts = Dataframe.columns.get_level_values("bodyparts")
    all_bpts = bpts.values[::3]
    bplist = bpts.unique().to_list()
    nbodyparts = len(bplist)
    # get color scheme
    colorclass = plt.cm.ScalarMappable(cmap=colormap)
    C = colorclass.to_rgba(np.linspace(0, 1, nbodyparts))
    colors = (C[:, :3] * 255).astype(np.uint8)
    # map bodyparts to color
    keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
    bpts2color = [(ind, colors[ind]) for ind in keep]
    
    # get list of images to render
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, _ = frame.shape
    # extract snippets longer than crit
    n, idx, lengths = repeating_numbers(labels)
    if choose_from_top_or_random != "random":
        # we don't filter anything out
        crit = -1

    rnges = [range(idx[i], idx[i] + j) 
                for i, j in enumerate(lengths) 
                if j >= crit]
    idx2 = [i 
            for i, j in enumerate(lengths) 
            if j >= crit]

    for i in tqdm(np.unique(labels)):
        # a: a set of range objects indicating which of those in rnges is for bout i
        a = [rng for j, rng in enumerate(rnges) if n[idx2[j]] == i]

        # choose either at random or the top 'counts' in length
        if choose_from_top_or_random == "random":
            generated_rnges = random.sample(a, min(len(a), counts))
            video_name_part = "example"
        else:
            # generated_rnges = sorted(
            #     a, key=lambda rng: len(rng), 
            #     reverse=True)[:min(len(a), counts)]
            generated_rnges = heapq.nlargest(
                min(len(a), counts), a, key=lambda rng: len(rng)
                )
            video_name_part = "top"
        
        # generate videos
        try:

            for k in range(len(generated_rnges)):
                video_name = 'group_{}_{}{}.mp4'.format(i, video_name_part, k+1)
                video = cv2.VideoWriter(os.path.join(subfolder, video_name), fourcc, output_fps, (width, height))
                for l in generated_rnges[k]:
                    image = images[l]
                    ### MAIN CHANGE ###
                    read_img = cv2.imread(os.path.join(frame_dir, image))
                    labeled_img = label_bodyparts_on_single_frame(read_img,
                                                                    index=l,
                                                                    x_filtered=df_x_filt, 
                                                                    y_filtered=df_y_filt,
                                                                    dotsize=dotsize,
                                                                    bpts2color=bpts2color,
                                                                    trailpoints=trailpoints)
                    video.write(labeled_img)
                    ### MAIN CHANGE END ###
                cv2.destroyAllWindows()
                video.release()
        except:
            traceback.print_exc() 
            print(f"Generating for label {i} failed...")
    return

def sort_nicely(l):
    def alphanum_key(s):
        def convert_int(s):
            return int(s) if s.isdigit() else s
        return [convert_int(c) for c in re.split('([0-9]+)', s)]
    l.sort(key=alphanum_key)


if __name__ == "__main__":
    FPS = 40
    FILTERING_NOISE_MAX_LENGTH = 5 # max length of noise filtered via filter_bouts_smaller_than_N_frames
    MIN_DESIRED_BOUT_LENGTH = 500
    COUNTS = 5
    OUTPUT_FPS = 30
    TRAILPOINTS = 0
    TOP_OR_RANDOM = "top" #"random"
    
    DOTSIZE = 7
    COLORMAP = "rainbow" # obtained from config.yaml on DLC side
    # we exclude "belly" as it isn't used to classify in this B-SOID
    BODYPARTS = ["snout",        "rightforepaw", "leftforepaw", 
                 "righthindpaw", "lefthindpaw",  "tailbase"] 
    
    FILE_OF_INTEREST = r"20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
    # r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
    LABELED_PREFIX = r"Feb-27-2024labels_pose_40Hz"

    OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\previous\B-SOID STUFF\BoutVideoBits\labeled_five_length_bout_filtered"
    # OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\previous\B-SOID STUFF\BoutVideoBits\labeled"
    
    FRAME_DIR     = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\pngs", FILE_OF_INTEREST.replace(".csv", ""))
    OUTPUT_PATH   = os.path.join(OUTPUT_FOLDER, FILE_OF_INTEREST.replace(".csv", ""))
    DATA_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv",      FILE_OF_INTEREST)
    LABELED_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\BSOID\Feb-27-2024", 
                                    LABELED_PREFIX + FILE_OF_INTEREST)
    
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