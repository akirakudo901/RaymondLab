# Author: Akira Kudo
# Created: 2024/04/12
# Last Updated: 2024/04/16

from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from dlc_io.utils import read_dlc_csv_file
from utils_to_be_replaced_oneday import bodypart_abbreviation_dict

def visualize_speed_of_bodypart_from_csv(
        csv_path : str,
        bodyparts : List[str],
        start : int,
        end : int,
        bodyparts2noise : Dict[str, np.ndarray]=None
        ):
    """
    Visualizes the speed of specified body parts between frames 
    start and end inclusive.
    If given with bodyparts2noise, can also visualize which frame 
    count as noise frame.

    :param str csv_path: Path to csv holding dlc data.
    :param List[str] bodyparts: List of body part names to visualize.
    :param int start: Frame number where to start visualization.
    :param int end: Frame number last shown in the visualization.
    :param Dict[str,np.ndarray] bodyparts2noise: Dictionary mapping 
    each body part to their identified noise frames, as boolean array.
    If given, which frame is noise is indicated for each body part.
    """
    df = read_dlc_csv_file(csv_path)
    existing_bodyparts = np.unique(df.columns.get_level_values('bodyparts')).tolist()
    concerned_bodyparts = [b for b in bodyparts if b in existing_bodyparts]
    if len(concerned_bodyparts) == 0: return

    # start visualization
    _, axes = plt.subplots(len(concerned_bodyparts), 1, figsize=(20, len(concerned_bodyparts)*3))
    for i, bpt in enumerate(concerned_bodyparts):
        ax = axes if (len(concerned_bodyparts) == 1) else axes[i]
    
        X, Y = df.loc[start:end, (bpt, 'x')], df.loc[start:end, (bpt, 'y')]
        diff = np.sqrt(np.square(np.diff(X)) + np.square(np.diff(Y)))
        padded_diff = np.insert(diff, 0, 0)

        # render the frames that were noise beneath
        if bodyparts2noise is not None and bpt in bodyparts2noise.keys():
            noise_bool = bodyparts2noise[bpt]
            _color_noise_frames(noise_bool_arr=noise_bool, 
                                max=np.max(diff).item(), 
                                min=np.min(diff).item(), 
                                axis=ax, start=start, color="red")
        
        # then also render the actual speeds
        ax.plot(range(start, end+1), padded_diff)
        ax.set_ylabel(bodypart_abbreviation_dict[bpt])
        if i < (len(concerned_bodyparts) - 1):
            ax.tick_params(axis='x', labelbottom=False)

    plt.suptitle("Body Parts Speed Over Time")
    plt.show()

def visualize_angle_of_bodypart_from_csv(
        csv_path : str,
        bodyparts : List[str],
        start : int,
        end : int,
        bodyparts2noise : Dict[str, np.ndarray]=None
        ):
    """
    Visualizes the angle of consecutive movements between specified 
    body parts, between frames start and end inclusive.
    If given with bodyparts2noise, can also visualize which frame 
    count as noise frame.

    :param str csv_path: Path to csv holding dlc data.
    :param List[str] bodyparts: List of body part names to visualize.
    :param int start: Frame number where to start visualization.
    :param int end: Frame number last shown in the visualization.
    :param Dict[str,np.ndarray] bodyparts2noise: Dictionary mapping 
    each body part to their identified noise frames, as boolean array.
    If given, which frame is noise is indicated for each body part.
    """
    # inner helpers
    def smaller_arc_angle(angle : np.ndarray):
        returned = np.copy(angle)
        complement_angle = 2*np.pi - angle
        where_complement_is_smaller = complement_angle < angle
        returned[where_complement_is_smaller] = angle[where_complement_is_smaller]
        return returned

    df = read_dlc_csv_file(csv_path)
    existing_bodyparts = np.unique(df.columns.get_level_values('bodyparts')).tolist()
    concerned_bodyparts = [b for b in bodyparts if b in existing_bodyparts]
    if len(concerned_bodyparts) == 0: return

    # start visualization
    _, axes = plt.subplots(len(concerned_bodyparts) * 2, 1, 
                           figsize=(20, len(concerned_bodyparts)*6))
    for i, bpt in enumerate(concerned_bodyparts):
        ax_angle, ax_angle_diff = axes[i*2], axes[i*2 + 1]
    
        X, Y = df.loc[start:end, (bpt, 'x')], df.loc[start:end, (bpt, 'y')]
        angle = np.arctan2(np.diff(Y), np.diff(X))
        padded_angle = np.insert(angle, 0, 0)
        # also compute and show angle difference
        angle_diff = smaller_arc_angle(np.abs(np.diff(padded_angle)))
        padded_angle_diff = np.insert(angle_diff, 0, 0)

        # render the frames that were noise beneath
        if bodyparts2noise is not None and bpt in bodyparts2noise.keys():
            noise_bool = bodyparts2noise[bpt]
            _color_noise_frames(noise_bool_arr=noise_bool, 
                                max=np.max(padded_angle).item(), 
                                min=np.min(padded_angle).item(), 
                                axis=ax_angle, start=start, color="red")
            _color_noise_frames(noise_bool_arr=noise_bool, 
                                max=np.max(padded_angle_diff).item(), 
                                min=np.min(padded_angle_diff).item(), 
                                axis=ax_angle_diff, start=start, color="red")
        
        # then also render the actual angles & their diffs
        ax_angle.plot(range(start, end+1), padded_angle)
        ax_angle.set_ylabel(bodypart_abbreviation_dict[bpt])
        
        ax_angle_diff.plot(range(start, end+1), padded_angle_diff)
        ax_angle_diff.set_ylabel(bodypart_abbreviation_dict[bpt])

        ax_angle.tick_params(axis='x', labelbottom=False)
        if i < (len(concerned_bodyparts) - 1):
            ax_angle_diff.tick_params(axis='x', labelbottom=False)

    plt.suptitle("Body Parts Angles Over Time")
    plt.show()


def visualize_property_of_bodypart_from_csv(
        csv_path : str,
        bodyparts : List[str],
        flag : int,
        start : int,
        end : int,
        bodyparts2noise : Dict[str, np.ndarray]=None
        ):
    """
    Visualizes four properties of consecutive movements:
    1) displacement; 2) angle; 3) change in angle; 4) acceleration.
    Which property to show is specified by 'flag'.
    
    This is done between specified body parts, between frames 
    'start' and 'end' inclusive.
    
    If given with bodyparts2noise, can also visualize which frame 
    count as noise frame.

    :param str csv_path: Path to csv holding dlc data.
    :param List[str] bodyparts: List of body part names to visualize.
    :param int flag: A flag indicating which property to visualize.
    Given { displacement : 0, angle : 1, angle_change : 2, acceleration : 3 },
    flag is the sum of 2^k for all k to be visualized.
    :param int start: Frame number where to start visualization.
    :param int end: Frame number last shown in the visualization.
    :param Dict[str,np.ndarray] bodyparts2noise: Dictionary mapping 
    each body part to their identified noise frames, as boolean array.
    If given, which frame is noise is indicated for each body part.
    """
    DISPLACEMENT, ANGLE, ANGLE_CHANGE, ACCELERATION = 'Displacement', 'Angle', 'Angle Change', 'Acceleration'
    flag_to_visual = {0:DISPLACEMENT, 1:ANGLE, 2:ANGLE_CHANGE, 3:ACCELERATION}

    # inner helpers
    def smaller_arc_angle(angle : np.ndarray):
        returned = np.copy(angle)
        complement_angle = 2*np.pi - angle
        where_complement_is_smaller = complement_angle < angle
        returned[where_complement_is_smaller] = angle[where_complement_is_smaller]
        return returned
    
    def process_flag(flag : int):
        returned = []; orig_flag = flag
        for i in sorted(flag_to_visual.keys(), reverse=True):
            if 2**i <= flag:
                returned.append(i)
                flag -= 2**i
        if flag != 0: raise Exception(f"flag {orig_flag} seems invalid ...")
        return [flag_to_visual[val] for val in returned]

    # process the flag
    to_render = process_flag(flag)
    if len(to_render) == 0: return

    # read data
    df = read_dlc_csv_file(csv_path)
    existing_bodyparts = np.unique(df.columns.get_level_values('bodyparts')).tolist()
    concerned_bodyparts = [b for b in bodyparts if b in existing_bodyparts]
    if len(concerned_bodyparts) == 0: return

    # start visualization
    _, axes = plt.subplots(len(concerned_bodyparts) * len(to_render), 1, 
                           figsize=(20, len(concerned_bodyparts)*len(to_render)*3))
    
    for bpt_idx, bpt in enumerate(concerned_bodyparts):
        X, Y = df.loc[start:end, (bpt, 'x')], df.loc[start:end, (bpt, 'y')]
        
        # for each visualization
        for vis_type_idx, vis in enumerate(to_render):
            if len(concerned_bodyparts) * len(to_render) == 1:
                ax = axes
            else:
                ax = axes[bpt_idx * len(to_render) + vis_type_idx]
            
            if vis == DISPLACEMENT:
                displacement = np.sqrt(np.diff(X)**2 + np.diff(Y)**2)
                to_plot = np.insert(displacement, 0, 0) # padding

            elif vis == ANGLE:
                angle = np.arctan2(np.diff(Y), np.diff(X))
                to_plot = np.insert(angle, 0, 0) # padding

            elif vis == ANGLE_CHANGE:
                angle = np.arctan2(np.diff(Y), np.diff(X))
                padded_angle = np.insert(angle, 0, 0) # padding
                angle_diff = smaller_arc_angle(np.abs(np.diff(padded_angle)))
                to_plot = np.insert(angle_diff, 0, 0) # padding

            elif vis == ACCELERATION:
                to_plot = None
                # TODO IMPLEMENT!
            
            # render the frames that were noise beneath
            if bodyparts2noise is not None and bpt in bodyparts2noise.keys():
                noise_bool = bodyparts2noise[bpt]
                _color_noise_frames(noise_bool_arr=noise_bool, 
                                    max=np.max(to_plot).item(), 
                                    min=np.min(to_plot).item(), 
                                    axis=ax, start=start, color="red")
                
            # then also render the actual angles & their diffs
            ax.plot(range(start, end+1), to_plot)
            ax.set_ylabel(bodypart_abbreviation_dict[bpt])
            
            if (bpt_idx < (len(concerned_bodyparts) - 1) or 
                vis_type_idx < (len(to_render) - 1)):
                ax.tick_params(axis='x', labelbottom=False)

    plt.suptitle(f"Body Parts {' & '.join(to_render)} Over Time")
    plt.show()


# HELPERS 
def _color_noise_frames(noise_bool_arr : np.ndarray, 
                        max : float, 
                        min : float, 
                        axis : matplotlib.axis.Axis, 
                        start : int, 
                        color="red"):
    """
    Given a plot axis where array 'also_plotted' is plotted,
    and a boolean array indicating which frames are noise, 
    color noise frames with the given color.

    :param np.ndarray noise_bool_arr: _description_
    :param float max: Maximum height we wanna paint.
    :param float min: Minimum height we wanna paint.
    :param matplotlib.axis.Axis axis: The Axis in question.
    :param int start: Starting frame number of the plot.
    :param str color: String indicating color of noise frames.
    """
    index_of_noise = np.where(noise_bool_arr)[0] + start
    axis.bar(index_of_noise, [max - min] * len(index_of_noise), 
                bottom=min, width=1, color=color)