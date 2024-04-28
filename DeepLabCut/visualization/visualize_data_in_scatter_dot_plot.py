# Author: Akira Kudo
# Created: 2024/04/26
# Last Updated: 2024/04/27

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEDIAN_LINESTYPE = '-'
MEAN_LINESTYPE = '--'

MOUSETYPE = 'mouseType'
TOTAL_DISTANCE_CM = 'totalDistanceCm'
CENTER_TIME = 'centerTime'

def visualize_data_in_scatter_dot_plot(csv_path : str,
                                        ylabel : str,
                                        y_val : str,
                                        colors : list,
                                        save_dir : str,
                                        save_name : str,
                                        x_val : str=MOUSETYPE,
                                        xlabel : str="Mouse Type",
                                        title : str=None,
                                        show_median : bool=False,
                                        show_mean : bool=False,
                                        save_figure : bool=True,
                                        show_figure : bool=True):
    """
    Takes in a data csv holding information obtained via running analysis.m
    onto multiple DLC csvs; and holding basic analysis info of open field
    mouse activity, such as mouseType, totalDistanceCm, centerTime etc.

    Plots the specified feature, aggregated by mouseType into a scatter plot.

    :param str csv_path: Path to csv holding DLC data.
    :param str ylabel: Label for y-axis.
    :param str y_val: Specifies what is rendered as y-label - has to be one 
    of columns in the DLC csv.
    :param list colors: A list of colors matching each x entries. If only one 
    is given, every column is of the same color.
    :param str save_dir: Directory for saving figure.
    :param str save_name: Name of saved figure.
    :param str x_val: Specifies what is rendered as x-label - has to be one
    of columns in the DLC csv. Defaults to MOUSETYPE.
    :param str xlabel: Label for x-axis, defaults to "Mouse Type".
    :param str title: Figure title, defaults to '{xlabel} per {ylabel}'.
    :param bool show_median: Whether to show median on graph, defaults to False.
    :param bool show_mean: Whether to show mean on graph, defaults to False.
    :param bool save_figure: Whether to save figure, defaults to True.
    :param bool show_figure: Whether to show figure, defaults to True.
    """
    df = pd.read_csv(csv_path)

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
    
    _, ax = plt.subplots(figsize=(3,6))
    for i, mstp in enumerate(unique_mousetypes):
        X = df[x_val].loc[df[MOUSETYPE] == mstp]
        Y = df[y_val].loc[df[MOUSETYPE] == mstp]
        ax.scatter(X, Y, color=colors[i])

        # set boundaries for mean & median bars
        barwidth = 1/3/(len(unique_mousetypes) + 1)
        xmax = (i+1) / (len(unique_mousetypes) + 1) - barwidth
        xmin = (i+1) / (len(unique_mousetypes) + 1) + barwidth
        # if show_mean is true, show mean
        if show_mean:
            ax.axhline(y=np.mean(Y), xmax=xmax, xmin=xmin,
                       color=colors[i], linestyle=MEAN_LINESTYPE)
        # if show_median is true, show median
        if show_median:
            ax.axhline(y=np.median(Y), xmax=xmax, xmin=xmin,
                       color=colors[i], linestyle=MEDIAN_LINESTYPE)
    
    # set the legend manually for median and mean
    median_line = mlines.Line2D([],[], color='black', 
                                linestyle=MEDIAN_LINESTYPE, label='Median')
    mean_line   = mlines.Line2D([],[], color='black', 
                                linestyle=MEAN_LINESTYPE,   label='Mean')
    plt.legend(handles=[median_line, mean_line])
    # set the other settingss
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.xticks(range(-1, len(unique_mousetypes)+1), rotation=45)
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
    CSV_NAME = "Q175_analysis_data_trunc_unfilt.csv"
    CSVFILE = os.path.join(CSV_DIR, CSV_NAME)

    MOUSE_GROUPNAME = "Q175"

    trunc_or_untrunc = CSV_NAME.replace('.csv', '').split('_')[-2]
    filter_or_unfiltered = CSV_NAME.replace('.csv', '').split('_')[-1]

    suffix = f"{MOUSE_GROUPNAME}_{filter_or_unfiltered}_{trunc_or_untrunc}"
    to_visualize = [
        [TOTAL_DISTANCE_CM, 'Total Distance (cm)', f"TotalDistanceCmPerMouseType_{suffix}.png"],
        [CENTER_TIME,       'Center Time (%)',     f"CenterTimePerMouseType_{suffix}.png"]
    ]

    # TIME_FRACTION_BY_QUADRANT : 'Time Fraction By Quadrant (%)', 
    # DISTANCE_BY_INTERVALS : 'Distance By Intervals (cm)'
    
    for visualized_var, var_name, fig_name in to_visualize:
        visualize_data_in_scatter_dot_plot(CSVFILE,
                                            y_val=visualized_var,
                                            xlabel=f'Mouse Type ({MOUSE_GROUPNAME})',
                                            ylabel=var_name,
                                            title=f"{var_name} Per Mouse Type",
                                            colors=['red', 'blue'],
                                            save_figure=True,
                                            save_dir=SAVE_DIR,
                                            save_name=fig_name,
                                            show_figure=False,
                                            show_mean=True,
                                            show_median=True)
    
   