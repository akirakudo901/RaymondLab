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
from skimage.draw import disk, line_aa
from tqdm import tqdm

# Given an image and corresponding coordinates for bodyparts, renders 
# the bodyparts onto the image, to be returned.
def label_bodyparts_on_single_frame(
    image : np.ndarray,
    index : int,
    Dataframe,
    pcutoff : float,
    dotsize,
    colormap,
    bodyparts2plot,
    trailpoints : int = 0
):
    """
    Creating individual frames with labeled body parts.
    image: np.ndarray
        The image we label with bodyparts.
    
    index: int
        The index of the image frame within the video.

    displayedbodyparts: list of strings, optional
        This selects the body parts that are plotted in the video. Either ``all``, 
        then all body parts from config.yaml are used or a list of strings that are 
        a subset of the full list.
        E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml 
        to select only these two body parts.

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

    bpts = Dataframe.columns.get_level_values("bodyparts")
    all_bpts = bpts.values[::3]
    
    ny, nx = image.shape[1], image.shape[0]

    df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T
    print(f"df_x: {df_x}")
    print(f"df_y: {df_y}")
    print(f"df_likelihood: {df_likelihood}")
    print(f"df_x.shape: {df_x.shape}")
    print(f"df_y.shape: {df_y.shape}")
    print(f"df_likelihood.shape: {df_likelihood.shape}")
    raise Exception()
    colorclass = plt.cm.ScalarMappable(cmap=colormap)

    bplist = bpts.unique().to_list()
    nbodyparts = len(bplist)
    map2bp = list(range(len(all_bpts)))
    
    keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
    bpts2color = [(ind, map2bp[ind]) for ind in keep]

    C = colorclass.to_rgba(np.linspace(0, 1, nbodyparts))
    colors = (C[:, :3] * 255).astype(np.uint8)

    with np.errstate(invalid="ignore"):
        for ind, num_bp in bpts2color:
            if df_likelihood[ind, index] > pcutoff:
                color = colors[num_bp]
                if trailpoints > 0:
                    for k in range(1, min(trailpoints, index + 1)):
                        draw_size_adjusted_disk(image, color, 
                            df_x[ind, index - k], df_y[ind, index - k], 
                            dotsize, ny, nx)
                draw_size_adjusted_disk(image, color, 
                    df_x[ind, index], df_y[ind, index], 
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
                                 pcutoff,
                                 dotsize,
                                 colormap,
                                 bodyparts2plot,
                                 trailpoints,
                                 choose_from_top_or_random : str="random"):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors (~300ms is advised)
    :param counts: scalar, number of randomly generated examples (~5 is advised)
    :param output_fps: integer, frame per second for the output video
    :param frame_dir: string, directory to where you extracted vid images
    :param output_path: string, directory to where you want to store short video examples

    :param choose_from_top_or_random: Whether to choose 'counts' groups at random or from the 
    top N 'counts' in terms of length. If not "random", chooses the top 'counts'.
    """
    print(f"Generating video snippets for: {data_csv_path}")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    subfolder = os.path.join(output_path, "random_n" if choose_from_top_or_random == "random" else "top_n")
    if not os.path.exists(subfolder):
        os.mkdir(subfolder)

    Dataframe = pd.read_csv(data_csv_path, index_col=0, header=[0,1,2], skip_blank_lines=False, low_memory=False)
    
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
                                                                    Dataframe=Dataframe,
                                                                    pcutoff=pcutoff,
                                                                    dotsize=dotsize,
                                                                    colormap=colormap,
                                                                    bodyparts2plot=bodyparts2plot,
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

def repeating_numbers(labels):
    """
    :param labels: 1D array, predicted labels
    :return n_list: 1D array, the label number
    :return idx: 1D array, label start index
    :return lengths: 1D array, how long each bout lasted for
    """
    i = 0
    n_list = []
    idx = []
    lengths = []
    while i < len(labels) - 1:
        n = labels[i]
        n_list.append(n)
        startIndex = i
        idx.append(i)
        while i < len(labels) - 1 and labels[i] == labels[i + 1]:
            i = i + 1
        endIndex = i
        length = endIndex - startIndex
        lengths.append(length)
        i = i + 1
    return n_list, idx, lengths

def extract_label_from_labeled_csv(labeled_csv_path):
    df = pd.read_csv(labeled_csv_path, low_memory=False)
    labels = df.loc[:,'B-SOiD labels'].iloc[2:].to_numpy()
    return labels


if __name__ == "__main__":
    FPS = 40
    MIN_DESIRED_BOUT_LENGTH = 500
    COUNTS = 5
    OUTPUT_FPS = 30
    TRAILPOINTS = 0
    TOP_OR_RANDOM = "random" #"top"
    
    PCUTOFF = 0
    DOTSIZE = 7
    COLORMAP = "rainbow" # obtained from config.yaml on DLC side
    # we exclude "belly" as it isn't used to classify in this B-SOID
    BODYPARTS = ["snout",        "rightforepaw", "leftforepaw", 
                 "righthindpaw", "lefthindpaw",  "tailbase"] 
    
    FILE_OF_INTEREST = r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
    LABELED_PREFIX = r"Feb-27-2024labels_pose_40Hz"

    OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\B-SOID STUFF\BoutVideoBits\labeled"
    
    FRAME_DIR     = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\pngs", FILE_OF_INTEREST.replace(".csv", ""))
    OUTPUT_PATH   = os.path.join(OUTPUT_FOLDER, FILE_OF_INTEREST.replace(".csv", ""))
    DATA_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv",      FILE_OF_INTEREST)
    LABELED_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\BSOID\Feb-27-2024", 
                                    LABELED_PREFIX + FILE_OF_INTEREST)

    labels = extract_label_from_labeled_csv(LABELED_CSV_PATH)

    create_labeled_behavior_bits(labels=labels, 
                                 crit=MIN_DESIRED_BOUT_LENGTH / FPS, 
                                 counts=COUNTS, 
                                 output_fps=OUTPUT_FPS, 
                                 frame_dir=FRAME_DIR, 
                                 output_path=OUTPUT_PATH, 
                                 data_csv_path=DATA_CSV_PATH,
                                 pcutoff=PCUTOFF,
                                 dotsize=DOTSIZE,
                                 colormap=COLORMAP,
                                 bodyparts2plot=BODYPARTS,
                                 trailpoints=TRAILPOINTS,
                                 choose_from_top_or_random=TOP_OR_RANDOM)









# A version that could deal with more complicated cases:
# def label_bodyparts_on_single_frame(
#     image : np.ndarray,
#     index : int,
#     Dataframe,
#     pcutoff : float,
#     dotsize,
#     colormap,
#     bodyparts2plot,
#     trailpoints : int=0,
#     cropping : bool=False,
#     x1 : int=0,
#     x2 : int=0,
#     y1 : int=0,
#     y2 : int=0,
#     bodyparts2connect,
#     skeleton_color,
#     draw_skeleton=False,
#     displaycropped,
#     color_by="bodypart"
# ):
#     """
#     Creating individual frames with labeled body parts.
#     image: np.ndarray
#         The image we label with bodyparts.
    
#     index: int
#         The index of the image frame within the video.

#     displayedbodyparts: list of strings, optional
#         This selects the body parts that are plotted in the video. Either ``all``, 
#         then all body parts from config.yaml are used or a list of strings that are 
#         a subset of the full list.
#         E.g. ['hand','Joystick'] for the demo Reaching-Mackenzie-2018-08-30/config.yaml 
#         to select only these two body parts.

#     draw_skeleton: bool
#         If ``True`` adds a line connecting the body parts making a skeleton on on 
#         each frame. The body parts to be connected and the color of these connecting 
#         lines are specified in the config file. By default: ``False``

#     trailpoints: int
#         Number of previous frames whose body parts are plotted in a frame (for displaying 
#         history). Default is set to 0.

#     displaycropped: bool, optional
#         Specifies whether only cropped frame is displayed (with labels analyzed therein), 
#         or the original frame with the labels analyzed in the cropped subset.

#     color_by : string, optional (default='bodypart')
#         Coloring rule. By default, each bodypart is colored differently.
#         If set to 'individual', points belonging to a single individual are colored the same.
#     """
#     bpts = Dataframe.columns.get_level_values("bodyparts")
#     all_bpts = bpts.values[::3]
#     if draw_skeleton:
#         color_for_skeleton = (
#             np.array(mcolors.to_rgba(skeleton_color))[:3] * 255
#         ).astype(np.uint8)
#         # recode the bodyparts2connect into indices for df_x and df_y for speed
#         bpts2connect = get_segment_indices(bodyparts2connect, all_bpts)

#     if displaycropped:
#         ny, nx = y2 - y1, x2 - x1
#     else:
#         ny, nx = image.shape[1], image.shape[0]

#     df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T
#     if cropping and not displaycropped:
#         df_x += x1
#         df_y += y1
#     colorclass = plt.cm.ScalarMappable(cmap=colormap)

#     bplist = bpts.unique().to_list()
#     nbodyparts = len(bplist)
#     if Dataframe.columns.nlevels == 3:
#         nindividuals = 1
#         map2bp = list(range(len(all_bpts)))
#         map2id = [0 for _ in map2bp]
#     else:
#         nindividuals = len(Dataframe.columns.get_level_values("individuals").unique())
#         map2bp = [bplist.index(bp) for bp in all_bpts]
#         nbpts_per_ind = (
#             Dataframe.groupby(level="individuals", axis=1).size().values // 3
#         )
#         map2id = []
#         for i, j in enumerate(nbpts_per_ind):
#             map2id.extend([i] * j)
#     keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
#     bpts2color = [(ind, map2bp[ind], map2id[ind]) for ind in keep]

#     if color_by == "bodypart":
#         C = colorclass.to_rgba(np.linspace(0, 1, nbodyparts))
#     else:
#         C = colorclass.to_rgba(np.linspace(0, 1, nindividuals))
#     colors = (C[:, :3] * 255).astype(np.uint8)

#     with np.errstate(invalid="ignore"):
#         if displaycropped:
#             image = image[y1:y2, x1:x2]

#         # Draw the skeleton for specific bodyparts to be connected as
#         # specified in the config file
#         if draw_skeleton:
#             for bpt1, bpt2 in bpts2connect:
#                 if np.all(df_likelihood[[bpt1, bpt2], index] > pcutoff) and not (
#                     np.any(np.isnan(df_x[[bpt1, bpt2], index]))
#                     or np.any(np.isnan(df_y[[bpt1, bpt2], index]))
#                 ):
#                     rr, cc, _ = line_aa(
#                         int(np.clip(df_y[bpt1, index], 0, ny - 1)),
#                         int(np.clip(df_x[bpt1, index], 0, nx - 1)),
#                         int(np.clip(df_y[bpt2, index], 1, ny - 1)),
#                         int(np.clip(df_x[bpt2, index], 1, nx - 1)),
#                     )
#                     image[rr, cc] = color_for_skeleton

#         for ind, num_bp, num_ind in bpts2color:
#             if df_likelihood[ind, index] > pcutoff:
#                 if color_by == "bodypart":
#                     color = colors[num_bp]
#                 else:
#                     color = colors[num_ind]
#                 if trailpoints > 0:
#                     for k in range(1, min(trailpoints, index + 1)):
#                         rr, cc = disk(
#                             (df_y[ind, index - k], df_x[ind, index - k]),
#                             dotsize,
#                             shape=(ny, nx),
#                         )
#                         image[rr, cc] = color
#                 rr, cc = disk(
#                     (df_y[ind, index], df_x[ind, index]), dotsize, shape=(ny, nx)
#                 )
#                 image[rr, cc] = color














# INITIAL ATTEMPT: MIGHT BE USEFUL LATER?
# def CreateVideoSlow(
#     image : np.ndarray,
#     index : int,
#     Dataframe,
#     tmpfolder,
#     dotsize,
#     colormap,
#     alphavalue,
#     pcutoff,
#     trailpoints,
#     cropping,
#     x1,
#     x2,
#     y1,
#     y2,
#     save_frames,
#     bodyparts2plot,
#     Frames2plot,
#     bodyparts2connect,
#     skeleton_color,
#     draw_skeleton,
#     displaycropped,
#     color_by,
# ):
#     """
#     Creating individual frames with labeled body parts.
#     :param np.ndarray image: The image we label with bodyparts.
#     :param int index: The index of the image frame within the video.
#     """


#     if displaycropped:
#         ny, nx = y2 - y1, x2 - x1
#     else:
#         ny, nx = image.shape[1], image.shape[0]

#     df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T
#     if cropping and not displaycropped:
#         df_x += x1
#         df_y += y1

#     bpts = Dataframe.columns.get_level_values("bodyparts")
#     all_bpts = bpts.values[::3]
#     if draw_skeleton:
#         bpts2connect = get_segment_indices(bodyparts2connect, all_bpts)

#     bplist = bpts.unique().to_list()
#     nbodyparts = len(bplist)
#     if Dataframe.columns.nlevels == 3:
#         nindividuals = 1
#         map2bp = list(range(len(all_bpts)))
#         map2id = [0 for _ in map2bp]
#     else:
#         nindividuals = len(Dataframe.columns.get_level_values("individuals").unique())
#         map2bp = [bplist.index(bp) for bp in all_bpts]
#         nbpts_per_ind = (
#             Dataframe.groupby(level="individuals", axis=1).size().values // 3
#         )
#         map2id = []
#         for i, j in enumerate(nbpts_per_ind):
#             map2id.extend([i] * j)
#     keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
#     bpts2color = [(ind,  [ind], map2id[ind]) for ind in keep]
#     if color_by == "individual":
#         colors = visualization.get_cmap(nindividuals, name=colormap)
#     else:
#         colors = visualization.get_cmap(nbodyparts, name=colormap)


#     # Prepare figure
#     prev_backend = plt.get_backend()
#     plt.switch_backend("agg")
#     dpi = 100
#     fig = plt.figure(frameon=False, figsize=(nx / dpi, ny / dpi))
#     ax = fig.add_subplot(111)

#     if cropping and displaycropped:
#         image = image[y1:y2, x1:x2]
#     ax.imshow(image)

#     if draw_skeleton:
#         for bpt1, bpt2 in bpts2connect:
#             if np.all(df_likelihood[[bpt1, bpt2], index] > pcutoff):
#                 ax.plot(
#                     [df_x[bpt1, index], df_x[bpt2, index]],
#                     [df_y[bpt1, index], df_y[bpt2, index]],
#                     color=skeleton_color,
#                     alpha=alphavalue,
#                 )

#     for ind, num_bp, num_ind in bpts2color:
#         if df_likelihood[ind, index] > pcutoff:
#             if color_by == "bodypart":
#                 color = colors(num_bp)
#             else:
#                 color = colors(num_ind)
#             if trailpoints > 0:
#                 ax.scatter(
#                     df_x[ind][max(0, index - trailpoints) : index],
#                     df_y[ind][max(0, index - trailpoints) : index],
#                     s=dotsize ** 2,
#                     color=color,
#                     alpha=alphavalue * 0.75,
#                 )
#             ax.scatter(
#                 df_x[ind, index],
#                 df_y[ind, index],
#                 s=dotsize ** 2,
#                 color=color,
#                 alpha=alphavalue,
#             )
#     ax.set_xlim(0, nx)
#     ax.set_ylim(0, ny)
#     ax.axis("off")
#     ax.invert_yaxis()
#     fig.subplots_adjust(
#         left=0, bottom=0, right=1, top=1, wspace=0, hspace=0
#     )
#     if save_frames:
#         fig.savefig(imagename)
#     ax.clear()

#     plt.switch_backend(prev_backend)