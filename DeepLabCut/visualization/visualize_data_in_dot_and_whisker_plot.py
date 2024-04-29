# Author: Akira Kudo
# Created: 2024/04/27
# Last Updated: 2024/04/28

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

MOUSETYPE = 'mouseType'
TIME_FRACTION_BY_QUADRANT = 'timeFractionByQuadrant'
DISTANCE_BY_INTERVALS = 'distanceByIntervals'

def visualize_data_in_scatter_dot_plot(csv_path : str,
                                        ylabel : str,
                                        x_vals : list,
                                        y_vals : list,
                                        colors : list,
                                        save_dir : str,
                                        save_name : str,
                                        xlabel : str="Mouse Type",
                                        title : str=None,
                                        save_figure : bool=True,
                                        show_figure : bool=True):
    """
    Takes in a data csv holding information obtained via running analysis.m
    onto multiple DLC csvs; and holding basic analysis info of open field
    mouse activity, such as mouseType, totalDistanceCm, centerTime etc.

    Plots the specified feature, aggregated by mouseType into a scatter plot.

    :param str csv_path: Path to csv holding DLC data.
    :param str ylabel: Label for y-axis.
    :param list x_vals: Specifies what is rendered at x-ticks in order.
    :param list y_vals: Specifies the columns that are rendered on the y-axis 
    in the given order - has to be matching columns in the DLC csv.
    :param list colors: A list of colors matching each x entries. If only one 
    is given, every column is of the same color.
    :param str save_dir: Directory for saving figure.
    :param str save_name: Name of saved figure.
    :param str xlabel: Label for x-axis, defaults to "Mouse Type".
    :param str title: Figure title, defaults to '{xlabel} per {ylabel}'.
    :param bool save_figure: Whether to save figure, defaults to True.
    :param bool show_figure: Whether to show figure, defaults to True.
    """
    df = pd.read_csv(csv_path)

    # make sure x_vals and y_vals are specified as pairs
    if len(x_vals) != len(y_vals):
        raise Exception("x_vals and y_vals have to be the same length...")
    unique_mousetypes = np.unique(df[MOUSETYPE])
    # make sure enough colors are specified unambiguously
    if len(colors) != 1 and len(colors) != len(unique_mousetypes):
        raise Exception("Given color have to match the number of mouse types in csv...")
    # otherwise if only one color is specified, set color to the same one
    if len(colors) == 1:
        colors = colors * len(unique_mousetypes)
    # set default title if not given
    if title is None:
        title = f'{xlabel} per {ylabel}'
    
    _, ax = plt.subplots()
    for i, mstp in enumerate(unique_mousetypes):
        # Y is calculated as mean of each specified columns as grouped by MOUSETYPE
        premean_Y = df[y_vals].loc[df[MOUSETYPE] == mstp]
        Y = premean_Y.mean(axis=0)
        # compute the standard error of the mean
        yerr = scipy.stats.sem(premean_Y, axis=0)
        # plot it
        ax.errorbar(x_vals, Y, fmt='-o', capsize=5, 
                    yerr=yerr, label=f'{mstp} n={premean_Y.shape[0]}', color=colors[i])
        ax.legend()
    
    # set the other settingss
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.tight_layout()

    if save_figure:
        plt.savefig(os.path.join(save_dir, save_name))
    
    if show_figure:
        plt.show()
    else:
        plt.close()
    
if __name__ == "__main__":
    SAVE_DIR = r"C:\Users\mashi\Desktop\temp\Q175\basic_analysis\fig"
    # SAVE_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\COMPUTED"
    
    CSV_DIR = r"C:\Users\mashi\Desktop\temp\Q175\basic_analysis"
    # CSV_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\previous\openfield\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
    CSV_NAME = "Q175_analysis_data_trunc_filt.csv"
    CSVFILE = os.path.join(CSV_DIR, CSV_NAME)

    MOUSE_GROUPNAME = "Q175"

    trunc_or_untrunc = CSV_NAME.replace('.csv', '').split('_')[-2]
    filter_or_unfiltered = CSV_NAME.replace('.csv', '').split('_')[-1]

    suffix = f"{MOUSE_GROUPNAME}_{filter_or_unfiltered}_{trunc_or_untrunc}"
    to_visualize = [
        [[TIME_FRACTION_BY_QUADRANT, 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], 
         ['1', '2', '3', '4'],
         "Quadrant number",
         'Time Fraction By Quadrant (%)', 
         f"TimeFractionByQuadrantPerMouseType_{suffix}.png"],
        [[DISTANCE_BY_INTERVALS, 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'],
         ['0~5','5~10','10~15','15~20','20~25','25~30'],
         "Interval (min)",
         'Distance By Intervals (cm)',     
         f"DistanceByIntervalsCmPerMouseType_{suffix}.png"]
    ]

    # TIME_FRACTION_BY_QUADRANT : 'Time Fraction By Quadrant (%)', 
    # DISTANCE_BY_INTERVALS : 'Distance By Intervals (cm)'
    
    for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize:
        visualize_data_in_scatter_dot_plot(CSVFILE,
                                            x_vals=x_vals,
                                            y_vals=visualized_var,
                                            xlabel=xlabel,
                                            ylabel=var_name,
                                            title=f"{var_name} Per Mouse Type",
                                            colors=['red', 'blue'],
                                            save_figure=True,
                                            save_dir=SAVE_DIR,
                                            save_name=fig_name,
                                            show_figure=False)