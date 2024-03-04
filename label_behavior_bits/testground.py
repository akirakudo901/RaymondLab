
# import os

# import cv2
# import pandas as pd

# from create_labeled_behavior_bits import label_bodyparts_on_single_frame


# FPS = 40
# MIN_DESIRED_BOUT_LENGTH = 200
# COUNTS = 5
# OUTPUT_FPS = 30
# TRAILPOINTS = 0

# PCUTOFF = 0
# DOTSIZE = 7
# COLORMAP = "rainbow" # obtained from config.yaml on DLC side
# # we exclude "belly" as it isn't used to classify in this B-SOID
# BODYPARTS = ["snout",        "rightforepaw", "leftforepaw", 
#                 "righthindpaw", "lefthindpaw",  "tailbase", "belly"] 

# FILE_OF_INTEREST = r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
# LABELED_PREFIX = r"Feb-27-2024labels_pose_40Hz"

# OUTPUT_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\B-SOID STUFF\BoutVideoBits\labeled"

# FRAME_DIR     = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\pngs", FILE_OF_INTEREST.replace(".csv", ""))
# OUTPUT_PATH   = os.path.join(OUTPUT_FOLDER, FILE_OF_INTEREST.replace(".csv", ""))
# DATA_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv",      FILE_OF_INTEREST)
# LABELED_CSV_PATH = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv\BSOID\Feb-27-2024", 
#                                 LABELED_PREFIX + FILE_OF_INTEREST)

# def extract_label_from_labeled_csv(labeled_csv_path):
#     df = pd.read_csv(labeled_csv_path, low_memory=False)
#     labels = df.loc[:,'B-SOiD labels'].iloc[2:].to_numpy()
#     return labels


# labels = extract_label_from_labeled_csv(LABELED_CSV_PATH)

# Dataframe = pd.read_csv(DATA_CSV_PATH, index_col=0, header=[0,1,2], skip_blank_lines=False, low_memory=False)

# l = 0

# image = os.listdir(FRAME_DIR)[0]

# read_img = cv2.imread(os.path.join(FRAME_DIR, image))
# cv2.imshow('Before', read_img) #TODO REMOVE
# cv2.waitKey(0)
# labeled_img = label_bodyparts_on_single_frame(read_img,
#                                             index=l,
#                                             Dataframe=Dataframe,
#                                             pcutoff=PCUTOFF,
#                                             dotsize=DOTSIZE,
#                                             colormap=COLORMAP,
#                                             bodyparts2plot=BODYPARTS,
#                                             trailpoints=TRAILPOINTS)
# cv2.imshow('After', labeled_img) #TODO REMOVE
# cv2.waitKey(0)

import os

import numpy as np
import pandas as pd

BRUTE_THRESHOLDING = True
FILE_OF_INTEREST = r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
data_csv_path = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv",      FILE_OF_INTEREST)

def adp_filt_bsoid_style(datax, datay, data_lh, brute_thresholding=False):
    datax_filt, datay_filt = np.zeros_like(datax), np.zeros_like(datay)
    
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float32))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        ##################
        # ADDED BY AKIRA
        if BRUTE_THRESHOLDING:
          llh = 0.8
        # print(f"x_llh : {x}, {llh}")
        ##################
        data_lh_float = data_lh[:, x].astype(np.float32)
        datax_filt[0, x], datay_filt[0, x] = datax[0, x], datay[0, x]
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                datax_filt[i, x], datay_filt[i, x] = datax_filt[i - 1, x], datay_filt[i - 1, x]
            else:
                datax_filt[i, x], datay_filt[i, x] = datax[i, x], datay[i, x]
    datax_filt = np.array(datax_filt).astype(np.float32)
    datay_filt = np.array(datay_filt).astype(np.float32)
    return datax_filt, datay_filt


from tqdm import tqdm
def adp_filt(currdf: object, pose):
    lIndex = []
    xIndex = []
    yIndex = []
    currdf = np.array(currdf[1:])
    for header in pose:
        if currdf[0][header + 1] == "likelihood":
            lIndex.append(header)
        elif currdf[0][header + 1] == "x":
            xIndex.append(header)
        elif currdf[0][header + 1] == "y":
            yIndex.append(header)
    curr_df1 = currdf[:, 1:]
    datax = curr_df1[1:, np.array(xIndex)]
    datay = curr_df1[1:, np.array(yIndex)]
    data_lh = curr_df1[1:, np.array(lIndex)]
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float32))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        ##################
        # ADDED BY AKIRA
        if BRUTE_THRESHOLDING:
          llh = 0.8
        # print(f"x_llh : {x}, {llh}")
        ##################
        data_lh_float = data_lh[:, x].astype(np.float32)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float32)
    return currdf_filt, perc_rect

n = 3

Dataframe = pd.read_csv(data_csv_path, index_col=0, header=[0,1,2], skip_blank_lines=False, low_memory=False)
df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T
df_x, df_y, df_likelihood = df_x.T, df_y.T, df_likelihood.T
data_x_filt1, data_y_filt1 = adp_filt_bsoid_style(df_x, df_y, df_likelihood, brute_thresholding=False)

Dataframe = pd.read_csv(data_csv_path, low_memory=False)
data_filt, _ = adp_filt(Dataframe, pose=list(range(7*3)))
data_filt = data_filt
data_x_filt2 = data_filt[:,0::2]
data_y_filt2 = data_filt[:,1::2]

print(f"data_x_filt1[:10]: {data_x_filt1[:n]}")
print(f"data_x_filt2[:10]: {data_x_filt2[:n]}")
print(f"data_y_filt1[:10]: {data_y_filt1[:n]}")
print(f"data_y_filt2[:10]: {data_y_filt2[:n]}")
print(f"data_x_filt1.shape: {data_x_filt1.shape}")
print(f"data_x_filt2.shape: {data_x_filt1.shape}")
print(f"data_y_filt1.shape: {data_y_filt1.shape}")
print(f"data_y_filt2.shape: {data_y_filt2.shape}")







# not_equal_idx = data_x_filt1 != data_x_filt2
# print(f"not_equal_idx: {np.where(not_equal_idx)}")
# print(f"unequal: {data_x_filt1[not_equal_idx]}")
# print(f"{data_x_filt2[not_equal_idx]}")
# print(f"unequal length: {len(not_equal_idx)}")
# print(f"y: {np.array_equal(data_y_filt1, data_y_filt2)}")

print(f"x: {np.allclose(data_x_filt1, data_x_filt2, atol=1e-3)}")
print(f"y: {np.allclose(data_y_filt1, data_y_filt2, atol=1e-3)}")