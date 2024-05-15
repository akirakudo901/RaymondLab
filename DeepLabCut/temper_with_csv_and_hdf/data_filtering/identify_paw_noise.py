# Author: Akira Kudo
# Created: 2024/03/11
# Last modified: 2024/03/20

import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlc_io.utils import read_dlc_csv_file

REST_CHANGE_TOLERANCE = 5 # variation of how many pixels is considered as non-moving
IMPOSSIBLE_MOVE_TOLERANCE = 70 # variation of how many pixels is considered as impossible movement

NOISE_TIMEOUT_FRAMENUM = 100

def identify_rest(dlc_csv_path : str, start : int, end : int):
    """
    Given a DLC prediction result, identify the mouse in rest.

    :param str dlc_csv_path: Path to csv that holds DLC prediction.
    :param int start: The start of sequence we wanna plot in frame number.
    :param int end: The end of sequence we wanna plot in frame number.
    """ 

    df = read_dlc_csv_file(dlc_csv_path)
    bpt_movement = []
    # check how bodypoints move over timepoints
    unique_bpts = np.unique(df.columns.get_level_values('bodyparts'))
    for unique_bpt in unique_bpts:        
        X, Y = df[unique_bpt, 'x'].to_numpy(), df[unique_bpt, 'y'].to_numpy()
        xy_diff = np.sqrt(np.square(np.diff(X)) + np.square(np.diff(Y)))
        where_bpt_did_not_move = xy_diff < REST_CHANGE_TOLERANCE
        bpt_movement.append(where_bpt_did_not_move)
    bpt_movement = np.array(bpt_movement)
    # then see which frames contains no moving bodypart
    no_bpt_moved = np.all(bpt_movement, axis=0)
    no_bpt_moved = np.insert(no_bpt_moved, 0, False) # pad first frame as not a rest frame

    if start > end: start, end = end, start
    start = min(max(start, 0), len(no_bpt_moved))
    end   = max(min(end, len(no_bpt_moved)),   0)
    
    # visualize where is a rest frame and where isn't
    print(f"There are {np.sum(no_bpt_moved)} frames with mouse at rest!")
    chosen_frames = no_bpt_moved[start:end]
    plt.step(range(len(chosen_frames)), chosen_frames)
    plt.title(f"Mouse at rest vs. others {start}~{end}")
    plt.show()

    return no_bpt_moved

def identify_bodypart_noise_in_rest(dlc_csv_path : str, bodypart : str, 
                            show_start : int, show_end : int, 
                            threshold : float=REST_CHANGE_TOLERANCE):
    """
    Given a DLC prediction result & bodypart name, identify any 
    frames in which only the bodypart moved abnormally.

    :param str dlc_csv_path: Path to csv that holds DLC prediction.
    :param str bodypart: The name of the bodypart examined for noise.
    :param float threshold: The number of pixels a bodypart has to travel
    over time in order to be deemed a noise. Defaults to CHANGE_TOLERANCE.

    :returns np.ndarray noise: An array indicating frames identified as noise.
    :returns np.ndarray wrong_frames: An array indicating frames identified as
    the wrong body part position (between every other frames).
    e.g. given noise: [0,0,1,0,0,1,0,0,1,0,1,0,1,0...], we get, 
        wrong_frames: [0,0,1,1,1,0,0,0,1,1,0,0,1,1...]
    """

    df = read_dlc_csv_file(dlc_csv_path)
    # condition 1: All bodyparts aren't moving but one paw.
    bpt_movement = []
    # check how bodypoints move over timepoints
    unique_bpts = np.unique(df.columns.get_level_values('bodyparts'))
    for unique_bpt in unique_bpts: 
        X, Y = df[unique_bpt, 'x'].to_numpy(), df[unique_bpt, 'y'].to_numpy()
        xy_diff = np.sqrt(np.square(np.diff(X)) + np.square(np.diff(Y)))
        where_bpt_did_not_move = xy_diff < threshold
        bpt_movement.append(where_bpt_did_not_move)
    bpt_movement = np.array(bpt_movement)
    
    # then see which frames contains no moving bodypart
    def nothing_but_bpt_of_interest_moves(bpt_of_interest):
        bpt_idx = unique_bpts.tolist().index(bpt_of_interest)
        other_bpt_movement = np.delete(bpt_movement, bpt_idx, axis=0)
        other_bpt_dont_move = np.all(other_bpt_movement, axis=0)
        bpt_of_interest_moved = np.invert(bpt_movement[bpt_idx, :])
        pawnoise_frame = np.logical_and(other_bpt_dont_move, bpt_of_interest_moved)
        pawnoise_frame = np.insert(pawnoise_frame, 0, False) # pad first frame as not a noise frame
        return pawnoise_frame
    
    # also see in which frame one of the paw move but all others are fixed
    no_bpt_moved = np.all(bpt_movement, axis=0)
    no_bpt_moved = np.insert(no_bpt_moved, 0, False) # pad first frame as not a rest frame
    
    noise = nothing_but_bpt_of_interest_moves(bodypart)

    # visualize where is a noise and where isn't
    print(f"There are {np.sum(noise)} frames with {bodypart} noise!")
    chosen_noise_frame = noise[show_start:show_end]
    plt.step(range(len(chosen_noise_frame)), chosen_noise_frame)
    plt.title(f"All frames where there is weird {bodypart} movement")
    plt.show()

    wrong_frames = find_wrong_bodypart_frame(noise)
    print("------------")
    print(f"There are {np.sum(wrong_frames)} frames where the {bodypart} is not in its correct position!")
    chosen_wrong_frame = wrong_frames[show_start:show_end]
    plt.step(range(len(chosen_wrong_frame)), chosen_wrong_frame)
    plt.title(f"All frames where the {bodypart} is wrong in position")
    plt.show()
    return noise, wrong_frames

def identify_bodypart_noise_by_impossible_speed(
        bpt_data : Union[str, pd.DataFrame],
        bodyparts : List[str],
        start : int, end : int, 
        savedir : str,
        threshold : float=IMPOSSIBLE_MOVE_TOLERANCE,
        save_figure : bool=True,
        show_figure : bool=True
        ):
    """
    Identifies noise of a given body part based on an impossible
    speed, specified as exceeding IMPOSSIBLE_MOVE_TOLERANCE.

    :param str or pd.DataFrame bpt_data: Either a path to csv holding dlc data, or a 
    pd.DataFrame holding the same data.
    :param List[str] bodypart: A list of body parts we examine for impossible move.
    :param int start: Beginning in frame of the filtering process - not deterministic
    depending on the frame, as we assume that the position of body parts in the 
    start frame is in a "correct" position. Play around with this to get desired result.
    :param int end: End of frames for the filtering process.
    :param str savedir: Directory to which we save figures.
    :param float threshold: Threshold which if exceeded indicates impossible movement.
    Defaults to IMPOSSIBLE_MOVE_TOLERANCE.
    :param bool save_figure: Whether to save figures, defaults to True.
    :param bool show_figure: Whether to show figures, defaults to True.

    :returns pd.DataFrame returned_df: A dataframe holding information about identified
    impossible frames & wrong frames based on those impossible frames.
    With Multiindex of form [all bodyparts] x ['imp','wrng','loc_wrng'], with column names 
    of ['bodyparts', 'content'].
    """
    if isinstance(bpt_data, str):
        df = read_dlc_csv_file(bpt_data, include_scorer=False)
    elif isinstance(bpt_data, pd.DataFrame):
        df = bpt_data
    else:
        raise Exception("bpt_data must be either a path to a DLC csv or a dataframe in the " + 
                        f"DLC format, but was instead of type {type(bpt_data)}...")
    
    data = []
    
    # check how bodypoints move over time
    unique_bpts = np.unique(df.columns.get_level_values('bodyparts'))
    considered_bpts = unique_bpts[np.isin(unique_bpts, np.array(bodyparts))]
    
    for bpt in considered_bpts:
        X = df.loc[start:end, (bpt, 'x')].to_numpy()
        Y = df.loc[start:end, (bpt, 'y')].to_numpy()
        xy_diff = np.sqrt(np.square(np.diff(X)) + np.square(np.diff(Y)))
        where_bpt_moved_impossibly = xy_diff >= threshold
        # pad first frame as not impossible
        where_bpt_moved_impossibly = np.insert(where_bpt_moved_impossibly, 0, False)
        data.append(where_bpt_moved_impossibly)
        
        # visualize where is a noise and where isn't
        print(f"There are {np.sum(where_bpt_moved_impossibly)} frames with {bpt} noise!")
        if show_figure or save_figure:
            plt.step(range(len(where_bpt_moved_impossibly)), where_bpt_moved_impossibly)
            plt.title(f"All frames where there is impossible {bpt} movement")
        if save_figure: plt.savefig(os.path.join(savedir, f'impossible_{bpt}_movement.png'))
        if show_figure: plt.show()
        else: plt.close()

        # find frames that are wrong based on the found impossible pattern
        wrong_frames = find_wrong_bodypart_frame(where_bpt_moved_impossibly)
        data.append(wrong_frames)
        print("------------")
        print(f"There are {np.sum(wrong_frames)} frames where the {bpt} is not in its correct position!")
        if show_figure or save_figure:
            plt.step(range(len(wrong_frames)), wrong_frames)
            plt.title(f"All frames where the {bpt} is wrong in position")
        if save_figure: plt.savefig(os.path.join(savedir, f'wrong_{bpt}_position.png'))
        if show_figure: plt.show()
        else: plt.close()

        # find frames that are wrong also incorporating location info
        loc_wrong_frames = find_wrong_bodypart_frame_also_from_location(
            array_of_noises=where_bpt_moved_impossibly, X=X, Y=Y,
            threshold=120, noise_timeout=15
            )
        data.append(loc_wrong_frames)
        print("------------")
        print(f"There are {np.sum(loc_wrong_frames)} frames where, location-based, " + 
              f"the {bpt} is not in its correct position!")
        if show_figure or save_figure:
            plt.step(range(len(loc_wrong_frames)), loc_wrong_frames)
            plt.title(f"All frames where the {bpt} is wrong based on location")
        if save_figure: plt.savefig(os.path.join(savedir, f'location_based_wrong_{bpt}_position.png'))
        if show_figure: plt.show()
        else: plt.close()
    
    data = np.array(data).T
    multiidx = pd.MultiIndex.from_product((considered_bpts, ['imp','wrng','loc_wrng']), 
                                          names=['bodyparts', 'content'])
    returned_df = pd.DataFrame(data, columns=multiidx)
    return returned_df

def identify_bodypart_noise_by_unprobable_movement(
        dlc_csv_path : str, bodyparts : List[str], 
        start : int, end : int
        ):
    """
    Identifies noise of a given body part based on an unprobable 
    movement, namely a jerk of movement into a different direction
    from the previous few frames' averages.

    :param str dlc_csv_path: Path to csv holding dlc data.
    :param List[str] bodypart: A list of body parts we examine for impossible move.
    :param int start: Beginning in frame of the filtering process.
    :param int end: End of frames for the filtering process.
    
    :returns pd.DataFrame returned_df: A dataframe holding information about identified
    unprobable frames & wrong frames based on those unprobable frames.
    With Multiindex of form [all bodyparts] x [unprobable_frames, wrong_frames].
    """
    # inner helpers
    def cart2pol(x, y):
        distance = np.sqrt(x**2 + y**2)
        angle = np.arctan2(y, x)
        return distance, angle
    
    def smaller_arc_angle(angle : np.ndarray):
        returned = np.copy(angle)
        complement_angle = 2*np.pi - angle
        where_complement_is_smaller = complement_angle < angle
        returned[where_complement_is_smaller] = angle[where_complement_is_smaller]
        return returned
    
    def plot_where_are_noise_on_axis(noise_bool_arr, also_plotted, axis):
        index_of_noise = np.where(noise_bool_arr)[0] + start
        maximum, minimum = np.max(also_plotted).item(), np.min(also_plotted).item()
        axis.bar(index_of_noise, [maximum - minimum] * len(index_of_noise), 
                 bottom=minimum, width=1, color="red")
    
    ANGLE_THRESHOLD, DISTANCE_THRESHOLD = np.pi/2, 0

    df = read_dlc_csv_file(dlc_csv_path, include_scorer=False)
    
    data = []
    
    # check how body parts move over time, in polar coordinate
    unique_bpts = np.unique(df.columns.get_level_values('bodyparts'))
    considered_bpts = unique_bpts[np.isin(unique_bpts, np.array(bodyparts))]
    
    for bpt in considered_bpts:
        X = df.loc[start:end, (bpt, 'x')].to_numpy()
        Y = df.loc[start:end, (bpt, 'y')].to_numpy()
        distance, angle = cart2pol(np.diff(X), np.diff(Y))
        angle_diff = smaller_arc_angle(np.abs(np.diff(angle)))
        # find where the body moved in an unprobable manner
        where_bpt_moved_unprobably = ((angle_diff   >    ANGLE_THRESHOLD) & 
                                      (distance[1:] > DISTANCE_THRESHOLD))
        # pad first two frames as not unprobable
        where_bpt_moved_unprobably = np.insert(where_bpt_moved_unprobably, 0, [False]*2)
        data.append(where_bpt_moved_unprobably)

        # visualize the angle & distance over time
        _, axes = plt.subplots(2,1)
        # first angle
        plot_where_are_noise_on_axis(where_bpt_moved_unprobably[2:], angle, axes[0])
        axes[0].plot(angle)
        axes[0].set_ylabel("Angle (Rad)")
        # then distance
        plot_where_are_noise_on_axis(where_bpt_moved_unprobably[2:], distance, axes[1])
        axes[1].plot(distance)
        axes[1].set_ylabel("Distance (pixel)"); axes[1].set_xlabel("Time (frame)")
        plt.suptitle(f"Angle and Distance over time for {bpt}")
        plt.show()

        # visualize where is a noise and where isn't
        print(f"There are {np.sum(where_bpt_moved_unprobably)} frames with {bpt} noise!")
        plt.step(range(len(where_bpt_moved_unprobably)), where_bpt_moved_unprobably)
        plt.title(f"All frames where there is unprobable {bpt} movement")
        plt.show()

        # TODO POSSIBLY FINISH!

    #     # find frames that are wrong based on the found impossible pattern
    #     wrong_frames = find_wrong_bodypart_frame(where_bpt_moved_impossibly)
    #     data.append(wrong_frames)
    #     print("------------")
    #     print(f"There are {np.sum(wrong_frames)} frames where the {bpt} is not in its correct position!")
    #     plt.step(range(len(wrong_frames)), wrong_frames)
    #     plt.title(f"All frames where the {bpt} is wrong in position")
    #     plt.show()
    
    # data = np.array(data).T
    # multiidx = pd.MultiIndex.from_product((considered_bpts, ['imp','wrng']), 
    #                                       names=['bodyparts', 'content'])
    # returned_df = pd.DataFrame(data, columns=multiidx)
    # return returned_df


def find_wrong_bodypart_frame(array_of_noises : np.ndarray):
    """
    Given an array of 'True' for each frame where bodypart is detected to 
    be noisy, create an array of hypothetical 'noisy labels' frames.
    e.g. given array1 indicating where bodypart moved irregularly:
            [0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0]
            Produce an estimate of where the bodypart is misplaced.
            [0,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1]
    :param np.ndarray array_of_noises: The array specifying where might be noise.
    Of shape [num_timepoints, ].
    :returns np.ndarray wrong_bodypart_frame: The array specifying where are
    'wrong' body part frames. Of shape [num_timepoints, ]
    """
    wrong_bodypart_frame = []
    frame_before_is_noise = False
    # for every frame in array_of_noises
    for i in range(len(array_of_noises)):
        current_frame_is_moving = array_of_noises[i]
        # if previous==noise & this==noise, back to non-noise mode
        if frame_before_is_noise and current_frame_is_moving:
            current_frame_is_noise = False
        # if previous==noise & this!=noise, this is still noise
        elif frame_before_is_noise and not current_frame_is_moving:
            current_frame_is_noise = True
        # if previous!=noise & this==noise, we are newly in a noise
        elif not frame_before_is_noise and current_frame_is_moving:
            current_frame_is_noise = True
        # if previous!=noise & this!=noise, this is not a noise either
        elif not frame_before_is_noise and not current_frame_is_moving:
            current_frame_is_noise = False
        
        wrong_bodypart_frame.append(current_frame_is_noise)
        frame_before_is_noise = current_frame_is_noise
    wrong_bodypart_frame = np.array(wrong_bodypart_frame)
    return wrong_bodypart_frame

def find_wrong_bodypart_frame_also_from_location(
        array_of_noises : np.ndarray,
        X : np.ndarray,
        Y : np.ndarray,
        threshold : float=REST_CHANGE_TOLERANCE,
        noise_timeout : int=NOISE_TIMEOUT_FRAMENUM
        ):
    """
    Given an array of 'True' for each frame where bodypart is detected to 
    be noisy, and checking location_data, create an array of hypothetical 
    'noisy labels' frames. 
    We check if after some flickers, the body part returns close to the 
    previous location (closeness determined relative to threshold). 
    If it does, we count the in-between as noise. If not, we discount the noise itself.
    Also, noise isn't counted as such if not going back within noise_timeout.

    :param np.ndarray array_of_noises: The array specifying where might be noise.
    Of shape [num_timepoints, ].
    :param np.ndarray X: X coordinates of body part, of shape [num_timepoints, ].
    :param np.ndarray Y: Y coordinates of body part, of shape [num_timepoints, ].
    :param float threshold: Let the latest non-noise frame location of bodypart A, 
    and its current location B. If we are on a noise frame, and dist(A, B) < threshold, 
    we consider the in-between frames as noise. Defaults to REST_CHANGE_TOLERANCE.
    :param int noise_timeout: How many frames before we stop looking to fix a noise.
    Defaults to NOISE_TIMEOUT_FRAMENUM.
    
    :returns np.ndarray wrong_bodypart_frame: The array specifying where are
    'wrong' body part frames. Of shape [num_timepoints, ]
    """
    wrong_bodypart_frame = np.full(array_of_noises.shape, False)
    in_noise, latest_nonnoise_idx = False, 0

    # for every frame in array_of_noises
    for i in range(len(array_of_noises)):
        current_frame_is_moving = array_of_noises[i]
        # 1. if in_noise & this==noise, check if the new location 
        # is close to the latest non-noise location
        if in_noise and current_frame_is_moving:
            x, y, prev_x, prev_y = X[i], Y[i], X[latest_nonnoise_idx], Y[latest_nonnoise_idx]
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            # if close enough, treat the in-between as noise
            if distance < threshold:
                wrong_bodypart_frame[latest_nonnoise_idx + 1 : i] = True
        # 2. if not in_noise & this==noise, we are newly in a noise
        elif not in_noise and current_frame_is_moving:
            in_noise = True
        # 3. if not in_noise & this!=noise, keep updating that this isn't noise either
        elif not in_noise and not current_frame_is_moving:
            latest_nonnoise_idx = i
        # 4. if in_noise & this!=noise, do nothing

        # also, time out if we've past noise_timeout frames from latest non-noise frame
        if in_noise and (i - latest_nonnoise_idx) >= noise_timeout:
            in_noise = False
        
    return wrong_bodypart_frame



if __name__ == "__main__":
    CSV_FOLDER = r"C:\Users\mashi\Desktop\RaymondLab\Experiments\B-SOID\Q175_Network\Q175_csv"
    #r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\B-SOID\Q175 Open Field CSVs\WT\snapshot950000"
    CSV_FILENAME = r"20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered.csv"
    #r"20220228203032_316367_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_950000.csv"

    CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILENAME)

    START, END = 0, 200

    pawnoise_frames, _ = identify_bodypart_noise_in_rest(CSV_PATH, 'rightforepaw', 
                                                 show_start=START, show_end=END)
    
    # run_values, run_starts, run_lengths = find_runs(pawnoise_frames)
    # pawnoise_lengths = run_lengths[run_values == True]
    # print(f"max: {np.max(pawnoise_lengths)}; min: {np.min(pawnoise_lengths)}")

    # START, END = 0, 500
    # identify_rest(CSV_PATH, start=START, end=END)

    identify_bodypart_noise_by_unprobable_movement(
        dlc_csv_path=CSV_PATH, 
        bodyparts='rightforepaw', 
        start=START, end=END
        )
