# Author: Akira Kudo
# Created: 2024/04/10
# Last Updated: 2024/04/11

import numpy as np
import pandas as pd

def filter_based_on_boolean_array(
    bool_arr : np.ndarray, 
    df : pd.DataFrame,
    filter_mode : str="latest"
    ):
    """
    Filter a given dataframe holding dlc body part data, where
    positions corresponding to True in bool_arr are considered
    noise frames. 
    Filter can be either to replace with the latest non-noise 
    position, or by linear interpolation between the position 
    of non-noise frames pinching the noise frames. 

    :param np.ndarray bool_arr: Array indicating which frame are noise.
    Shape of [num_timepoints].
    :param pd.DataFrame df: Dataframe containing DLC body part data.
    Shape of [num_timepoints x {num_bodypoints x (x,y,likelihood)}].
    :param str filter_mode: How to replace filtered data. 
    * "latest": picking the latest non-noise position
    * "linear": linear interpolation of non-noise frames pinching noises
    Defaults to "latest".
    """
    FILTER_MODES = ["latest", "linear"]
    if filter_mode not in FILTER_MODES:
        raise Exception(f"filter_mode must be one of {','.join(FILTER_MODES)} ...")
    # bool_arr and first dimension of df must be equal
    if bool_arr.shape[0] != df.shape[0]:
        raise Exception("bool_arr and df must have the save first dimension, but " + 
                        f"had {bool_arr.shape[0]} and {df.shape[0]} respectively...")
    
    returned_df = df.copy(deep=True)
    # filter_mode = latest case
    if filter_mode == "latest":
        for curr_frame_idx in range(1, len(bool_arr)):
            is_noise = bool_arr[curr_frame_idx]
            # if we are at a noise frame, copy the previous frame entry
            if is_noise:
                returned_df.loc[curr_frame_idx, :] = returned_df.loc[curr_frame_idx - 1, :]
        return returned_df
    # filter_mode = linear case
    elif filter_mode == "linear":
        # keep track of the last non-noise row
        last_nonnoise_row_idx = 0
        for curr_frame_idx in range(1, len(bool_arr)):
            is_noise, prev_was_noise = bool_arr[curr_frame_idx], bool_arr[curr_frame_idx-1]
            # if at a non-noise frame & previous frame was noise, we linear interpolate
            if not is_noise and prev_was_noise:
                # find the stride per timepoint for the noise window
                noise_window_size = curr_frame_idx - last_nonnoise_row_idx - 1
                change_per_stride = (df.loc[curr_frame_idx,:] - df.loc[last_nonnoise_row_idx,:]) / (noise_window_size + 1)
                # apply the interpolation
                for curr_row_idx in range(last_nonnoise_row_idx+1, curr_frame_idx):
                    stride_size = curr_row_idx - last_nonnoise_row_idx
                    returned_df.loc[curr_row_idx,:] = (returned_df.loc[last_nonnoise_row_idx,:] + 
                                                       change_per_stride * stride_size)
            # if this frame is not a noise, update last_nonnoise_row_idx
            if not is_noise:
                last_nonnoise_row_idx = curr_frame_idx

        return returned_df

if __name__ == "__main__":
    df = pd.DataFrame([[0, 0,  0],
                       [1,10,100],
                       [9,54,876],
                       [9,53,870],
                       [8,54,875],
                       [5,50,500],
                       [2,45,238],
                       [7,70,700]])
    bool_arr = np.array([False,False,True,True,True,False,True,False])
    latest_filtered = filter_based_on_boolean_array(
        bool_arr=bool_arr, df=df, filter_mode="latest")
    linear_filtered = filter_based_on_boolean_array(
        bool_arr=bool_arr, df=df, filter_mode="linear")
    print(f"linear_filtered: \n{latest_filtered}")
    print(f"linear_filtered: \n{linear_filtered}")
    