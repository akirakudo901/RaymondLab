# Author: Akira Kudo
# Created: 2024/04/12
# Last Updated: 2024/04/12

from typing import Dict, List

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
            where_noise_are = np.where(noise_bool)[0] + start
            ax.bar(where_noise_are, [np.max(diff).item()] * len(where_noise_are), 
                   width=1, color="red")
        
        # then also render the actual speeds
        ax.plot(range(start, end+1), padded_diff)
        ax.set_ylabel(bodypart_abbreviation_dict[bpt])
        if i < (len(concerned_bodyparts) - 1):
            ax.tick_params(axis='x', labelbottom=False)

    plt.suptitle("Body Parts Speed Over Time")
    plt.show()