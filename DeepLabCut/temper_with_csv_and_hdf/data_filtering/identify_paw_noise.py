# Author: Akira Kudo
# Created: 2024/03/11
# Last modified: 2024/03/20

import os

import matplotlib.pyplot as plt
import numpy as np

from dlc_io.utils import read_dlc_csv_file

REST_CHANGE_TOLERANCE = 5 # variation of how many pixels is considered as non-moving
IMPOSSIBLE_MOVE_TOLERANCE = 50 # variation of how many pixels is considered as impossible movement

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
