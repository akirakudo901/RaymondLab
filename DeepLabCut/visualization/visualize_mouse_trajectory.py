# Author: Akira Kudo
# Created: 2024/04/27
# Last Updated: 2024/05/19
# Created: -
# Last updated: 2024/05/19

import os

import matplotlib.pyplot as plt
import numpy as np

from dlc_io.utils import read_dlc_csv_file

FIG_X_MAX, FIG_X_MIN, FIG_Y_MAX, FIG_Y_MIN = 1080, 0, 1080, 0

def visualize_mouse_trajectory(csvpath : str, 
                               figureName : str,
                               save_path : str,
                               start : int=0, 
                               end : int=None,
                               bodypart : str="tailbase",
                               show_figure : bool=True,
                               save_figure : bool=True):
    """
    Plots how the mouse moved in the cage over time, with points progressively
    increasing in shade darkness for later time points. The raw data from the 
    DLC csvs is directly used to make the plot (unlike BSOID side's 
    plot_mouse_trajectory which uses smoothed data).

    :param str csvpath: Path to csv holding DLC data.
    :param str figureName: Name of figure to be saved.
    :param str save_path: Path to where we save the figure.
    :param int start: Start frame from which we render, defaults to 0
    :param int end: End frame up to which we render, defaults to end of video.
    :param str bodypart: Which body part to plot, defaults to "tailbase"
    :param bool show_figure: Whether to show figures as we go, defaults to True
    :param bool save_figure: Whether to save figures as we go, defaults to True
    """
    raw_data = read_dlc_csv_file(csvpath, include_scorer=False)
    # bodyparts in order: snout, rightforepaw, leftforepaw, righthindpaw,
    #                     lefthindpaw, tailbase, belly
    if bodypart.lower() not in np.unique(raw_data.columns.get_level_values('bodyparts')):
        bodypart = "tailbase"
        print("Given bodypart is unexpected - using tailebase instead!")
    # get the x/y data for the specified body part
    X, Y = raw_data[bodypart, 'x'].to_numpy(), raw_data[bodypart, 'y'].to_numpy()

    if end is None: end = len(X)
    plot_figure_over_time(X, Y, save_path, start, end, figureName, show_figure, save_figure)

# Plots joy stick position over time, with points progressively increasing in
# shade darkness for later time points
def plot_figure_over_time(X : np.ndarray, Y : np.ndarray,
                          save_path : str,
                          start : int, end : int, figureName : str,
                          show_figure : bool=True,
                          save_figure : bool=True):
    # clip end
    end = min(end, len(X))
    # render chunks of the image at once for speed up
    num_rendering_chunk = 1000
    chunk_size = (end - start) // 1000
    blueColor = plt.cm.Blues(np.linspace(0.1,1,num_rendering_chunk))

    fig, ax = plt.subplots(figsize=(6,6))
    for k in range(num_rendering_chunk):
        render_start = start + chunk_size * k
        render_end   = start + chunk_size * (k + 1) - 1
        render_end = min(render_end, end)
        ax.plot(X[render_start:render_end], Y[render_start:render_end],
                color=blueColor[k])
    plt.xlabel('X'); plt.ylabel('Y')
    plt.title(figureName)
    plt.xlim(FIG_X_MIN, FIG_X_MAX); plt.ylim(FIG_Y_MIN, FIG_Y_MAX)

    if save_figure:
        if not os.path.exists(save_path):
            print(f"{os.path.basename(save_path)} did not exist - created it!")
            os.mkdir(save_path)
        print(f"Saving {figureName} to {save_path}!")
        plt.savefig(os.path.join(save_path, figureName))

    if show_figure: plt.show()
    else: plt.close()