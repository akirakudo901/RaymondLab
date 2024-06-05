# Author: Akira Kudo
# Created: 2024/05/31
# Last Updated: 2024/06/05

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILENAME = 'fileName'
MOUSETYPE = 'mouseType'
CENTERTIME_BY_INTERVALS = "centerTimeByIntervals"
TIME_FRACTION_BY_QUADRANT = 'timeFractionByQuadrant'
DISTANCE_BY_INTERVALS = 'distanceByIntervals'
DISTANCE_BY_INTERVALS_NORM = 'distanceByIntervals_norm'

RENAMED_DISTBYINTER_NORM = {
    DISTANCE_BY_INTERVALS_NORM : f"{DISTANCE_BY_INTERVALS} Normalized (0~5)", 
    'Unnamed: 15_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (5~10)",
    'Unnamed: 16_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (10~15)",
    'Unnamed: 17_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (15~20)",
    'Unnamed: 18_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (20~25)",
    'Unnamed: 19_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (25~30)",
    }

def visualize_individual_timeseries_data_from_csv(
        csv_path : str,
        ylabel : str,
        x_vals : list,
        y_vals : list,
        colors : list,
        save_dir : str,
        save_name : str,
        xlabel : str="Mouse Type",
        title : str=None,
        save_figure : bool=True,
        show_figure : bool=True
        ):
    """
    Takes in a data csv holding information obtained via running analysis.m
    onto multiple DLC csvs; and holding basic analysis info of open field
    mouse activity, such as mouseType, totalDistanceCm, centerTime etc.

    Plots the specified feature for each individual mouse, colored based
    on mouseType into a time series plot.

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
    visualize_individual_timeseries_data_from_dataframe(
        df=df, ylabel=ylabel, x_vals=x_vals, y_vals=y_vals,
        colors=colors, save_dir=save_dir, save_name=save_name,
        xlabel=xlabel, title=title, save_figure=save_figure,
        show_figure=show_figure
        )


def visualize_individual_timeseries_data_from_dataframe(
        df : pd.DataFrame,
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
    Takes in a dataframe holding information obtained via running analysis.m
    onto multiple DLC csvs; and holding basic analysis info of open field
    mouse activity, such as mouseType, totalDistanceCm, centerTime etc.

    Plots the specified feature for each individual mouse, colored based
    on mouseType into a time series plot.

    :param pd.DataFrame df: Dataframe holding DLC csv data.
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
    
    _, axes = plt.subplots(1, len(unique_mousetypes), figsize=(12,6))
    # plot each mouse
    for i, mstp in enumerate(unique_mousetypes):
        ax = axes if (len(unique_mousetypes) == 1) else axes[i]
        
        all_Y = df[y_vals].loc[df[MOUSETYPE] == mstp]
        for idx, row in all_Y.iterrows():
            mousename = df.loc[idx, FILENAME]
            ax.plot(x_vals, row, marker='o', linestyle='solid', label=mousename)
        
        ax.legend()
        ax.set_xlabel(f'{xlabel} ({mstp})')
        if i == 0:
            ax.set_ylabel(ylabel)

    # set the other settingss
    plt.suptitle(title)
    plt.tight_layout()

    if save_figure:
        plt.savefig(os.path.join(save_dir, save_name))
    
    if show_figure:
        plt.show()
    else:
        plt.close()

def normalize_columns_relative_to_entry(df : pd.DataFrame, 
                                        columns : list,
                                        normalize_to : str):
    """
    Takes a data frame with numerical data in 'columns',
    normalizing them such that values in 'normalize_to' is 1.
    The newly normalized columns are appended at the end of the 
    data frame and returned as a new data frame.

    :param pd.DataFrame df: Data frame holding data to normalize.
    :param list columns: Columns to be normalized according to normalize_to.
    :param str normalize_to: Column against which 'columns' are normalized.
    :return pd.DataFrame norm_df: A new data frame as concatenation of df
    and new columns, renamed 'normalize_to'_norm, behind it.
    """
    columns_in_df = np.isin(columns, df.columns)
    if not np.all(columns_in_df): 
        raise Exception("These columns aren't part of df.columns: " + 
                        f"{np.array(columns)[np.logical_not(columns_in_df)]}; not in " + 
                        f"{df.columns}...")
    if not np.isin(normalize_to, df.columns):
        raise Exception("Column 'normalize_to' must be part of df.columns but isn't: " + 
                        f"{normalize_to}; not in {df.columns}...")
    
    normalized = df.loc[:, columns].div(df[normalize_to], axis=0)
    normalized.columns = normalized.columns.astype(str) + '_norm'
    norm_df = pd.concat([df, normalized], axis=1)
    
    return norm_df

def normalize_distanceByIntervals(df : pd.DataFrame, 
                                  savedir : str=None,
                                  savename : str=None):
    """
    Specifically normalizes the 'distanceByIntervals' entry of a typical data
    frame obtained from running basic analysis on DLC data.
    If given with savedir & savename, save the normalized data frame at savedir
    with name savename.
    """
    # normalize the data frame
    to_normalize = ['distanceByIntervals', 'Unnamed: 15', 'Unnamed: 16', 
                    'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19']
    normalize_to = 'distanceByIntervals'
    norm_df = normalize_columns_relative_to_entry(df=df, columns=to_normalize, normalize_to=normalize_to)
    # rename columns
    norm_df.columns = [RENAMED_DISTBYINTER_NORM.get(e, e) for e in norm_df.columns.tolist()]
    # if given with savename & savedir, save
    if savedir is not None and savename is not None:
        print(f"Saving to {savedir}!")
        norm_df.to_csv(os.path.join(savedir, savename))
    return norm_df
    
if __name__ == "__main__":

    def visualize_distance_and_centertime_by_intervals_from_csv(
            csvfile : str,
            groupname : str,
            truncated : bool,
            filtered : bool,
            save_dir : str,
            save_figure : bool,
            show_figure : bool, 
            colors : list):
        
        df = pd.read_csv(csvfile)
        
        visualize_distance_and_centertime_by_intervals_from_dataframe(
            df=df, groupname=groupname, truncated=truncated,
            filtered=filtered, save_dir=save_dir, save_figure=save_figure,
            show_figure=show_figure, colors=colors)
                
    def visualize_distance_and_centertime_by_intervals_from_dataframe(
            df : pd.DataFrame,
            groupname : str,
            truncated : bool,
            filtered : bool,
            save_dir : str,
            save_figure : bool,
            show_figure : bool, 
            colors : list):
        filter_or_unfiltered = "filtered" if filtered else "unfiltered"
        trunc_or_untrunc = "trunc" if truncated else "untrunc"
        suffix = f"{groupname}_{filter_or_unfiltered}_{trunc_or_untrunc}"

        to_visualize_individuals_as_timeseries = [
            # [[DISTANCE_BY_INTERVALS, 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19'],
            # ['0~5','5~10','10~15','15~20','20~25','25~30'],
            # "Interval (min)",
            # 'Distance By Intervals (cm)',     
            # f"DistanceByIntervalsCmPerIndividualMouse_{suffix}.png"],
            # [[CENTERTIME_BY_INTERVALS, 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9'],
            # ['0~5','5~10','10~15','15~20','20~25','25~30'],
            # "Interval (min)",
            # 'Center Time By Intervals (%)',
            # f"CenterTimeByIntervalsPercPerIndividualMouse_{suffix}.png"],
            [list(RENAMED_DISTBYINTER_NORM.values()),
            ['0~5','5~10','10~15','15~20','20~25','25~30'],
            "Interval (min)",
            'Distance By Intervals Normalized To First Interval (ratio)',     
            f"DistanceByIntervalsNormCmPerIndividualMouse_{suffix}.png"],
        ]

        for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_individuals_as_timeseries:
                visualize_individual_timeseries_data_from_dataframe(df,
                                                    x_vals=x_vals,
                                                    y_vals=visualized_var,
                                                    xlabel=xlabel,
                                                    ylabel=var_name,
                                                    title=f"{var_name} Per Mouse Type ({groupname})",
                                                    colors=colors,
                                                    save_figure=save_figure,
                                                    save_dir=save_dir,
                                                    save_name=fig_name,
                                                    show_figure=show_figure)

    
    if False:
        MOUSE_GROUPNAME = "Q175"
        SAVE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\{}\fig".format(MOUSE_GROUPNAME)
        
        CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
        CSV_PATHS = [
            os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_Q175_analysis_data_filt.csv"),
            os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_Q175_analysis_data_unfilt.csv"),
        ]

        colors = ['red', 'blue']

        for csv in CSV_PATHS:
            visualize_distance_and_centertime_by_intervals_from_csv(csvfile=csv, 
                                                           groupname=MOUSE_GROUPNAME,
                                                           truncated=False,
                                                           filtered='unfilt' not in csv,
                                                           save_dir = SAVE_DIR,
                                                           save_figure=True,
                                                           show_figure=True,
                                                           colors=colors)
    
    if False:
        MOUSE_GROUPNAME = "YAC128"
        SAVE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\{}\fig".format(MOUSE_GROUPNAME)
        
        CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
        CSV_PATHS = [
            os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_YAC128_analysis_data_filt.csv"),
            os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_YAC128_analysis_data_unfilt.csv"),
        ]

        colors = ['black', 'pink']

        for csv in CSV_PATHS:
            visualize_distance_and_centertime_by_intervals_from_csv(csvfile=csv, 
                                                           groupname=MOUSE_GROUPNAME,
                                                           truncated=False,
                                                           filtered='unfilt' not in csv,
                                                           save_dir = SAVE_DIR,
                                                           save_figure=True,
                                                           show_figure=True,
                                                           colors=colors)
    
    # normalized distance by interval
    if True:
        for mouse_groupname in [
            "YAC128", 
            "Q175"
        ]:
            SAVE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\{}\fig".format(mouse_groupname)
        
            CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
            CSV_PATHS = [
                os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_{}_analysis_data_filt.csv".format(mouse_groupname)),
                os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_{}_analysis_data_unfilt.csv".format(mouse_groupname)),
            ]

            colors = ['black', 'pink']

            for csv in CSV_PATHS:
                df = pd.read_csv(csv)
                norm_df = normalize_distanceByIntervals(df, savedir=CSV_FOLDER, savename=os.path.basename(csv))

                visualize_distance_and_centertime_by_intervals_from_dataframe(
                    df=norm_df,
                    groupname=mouse_groupname,
                    truncated=False,
                    filtered='unfilt' not in csv,
                    save_dir=SAVE_DIR,
                    save_figure=True,
                    show_figure=True,
                    colors=colors)