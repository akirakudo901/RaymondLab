import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from .preprocessing import adp_filt_bsoid_style

BRUTE_THRESHOLDING = False
FILE_OF_INTEREST = r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_500000.csv"
data_csv_path = os.path.join(r"D:\B-SOID\Leland B-SOID YAC128 Analysis\Q175\WT\csv",      FILE_OF_INTEREST)

# original B-SOID filter code
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

if __name__ == "__main__":
    n = 3

    Dataframe = pd.read_csv(data_csv_path, index_col=0, header=[0,1,2], skip_blank_lines=False, low_memory=False)
    df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T
    df_x, df_y, df_likelihood = df_x.T, df_y.T, df_likelihood.T
    data_x_filt1, data_y_filt1 = adp_filt_bsoid_style(df_x, df_y, df_likelihood, brute_thresholding=BRUTE_THRESHOLDING)

    Dataframe = pd.read_csv(data_csv_path, low_memory=False)
    data_filt, _ = adp_filt(Dataframe, pose=list(range(7*3)))
    data_filt = data_filt
    data_x_filt2 = data_filt[:,0::2]
    data_y_filt2 = data_filt[:,1::2]

    print(f"data_x_filt1[:{n}]: {data_x_filt1[:n]}")
    print(f"data_x_filt2[:{n}]: {data_x_filt2[:n]}")
    print(f"data_y_filt1[:{n}]: {data_y_filt1[:n]}")
    print(f"data_y_filt2[:{n}]: {data_y_filt2[:n]}")
    print(f"data_x_filt1.shape: {data_x_filt1.shape}")
    print(f"data_x_filt2.shape: {data_x_filt1.shape}")
    print(f"data_y_filt1.shape: {data_y_filt1.shape}")
    print(f"data_y_filt2.shape: {data_y_filt2.shape}")

    print(f"x: {np.allclose(data_x_filt1, data_x_filt2, atol=1e-3)}")
    print(f"y: {np.allclose(data_y_filt1, data_y_filt2, atol=1e-3)}")