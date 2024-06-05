# Author: Akira Kudo
# Created: 2024/06/03
# Last Updated: 2024/06/04

import os

from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm

def visualize_qq_plot_from_csv(csv_path : str,
                                columns : list,
                                save_dir : str,
                                save_prefix : str,
                                mousetype : str,
                                dist=None,
                                dist_name : str="Standard Normal",
                                line : str=None, 
                                fit : bool=False,
                                save_figure : bool=True,
                                show_figure : bool=True):
    """
    Plots a Q-Q plot of a given column in a data frame against a specified
    distribution which defaults to the standard normal. 
    The data frame is stored in the given csv path.

    :param str csv_path: Path to csv holding DLC data.
    :param list columns: Name of columns for which we render the Q-Q plots.
    :param str save_dir: Directory for saving figure.
    :param str save_prefix: Prefix to a saved figure, which will then include 
    additional info on which column is rendered. 
    :param str mousetype: Type of mouse for figure title purpose.
    :param dist: Distribution to render, defaults to standard normal 
    distribution. Check following for more detail:
    https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html
    :param str dist_name: Name of distribution we compare sample against, for figure title.
    Defaults to default, standard normal.
    :param str line: What line to render for the Q-Q plot, defaults to None.
    :param bool fit: Whether to fit the sample distribution to 'dist' before plotting, 
    thereby attempting to make it into a 45 degree line. Defaults to False.
    :param bool save_figure: Whether to save figure, defaults to True.
    :param bool show_figure: Whether to show figure, defaults to True.
    """
    df = pd.read_csv(csv_path)
    
    visualize_qq_plot_from_dataframe(
        df=df, columns=columns, save_dir=save_dir, save_prefix=save_prefix, 
        mousetype=mousetype, dist=dist, dist_name=dist_name, line=line, fit=fit, 
        save_figure=save_figure, show_figure=show_figure
        )

def visualize_qq_plot_from_dataframe(df : pd.DataFrame,
                                    columns : list,
                                    save_dir : str,
                                    save_prefix : str,
                                    mousetype : str,
                                    dist=None,
                                    dist_name : str="Standard Normal",
                                    line : str=None, 
                                    fit : bool=False,
                                    save_figure : bool=True,
                                    show_figure : bool=True):
    """
    Plots a Q-Q plot of a given column in a data frame against a specified
    distribution which defaults to the standard normal.

    :param pd.DataFrame df: Data frame holding DLC data.
    :param list columns: Name of columns for which we render the Q-Q plots.
    :param str save_dir: Directory for saving figure.
    :param str save_prefix: Prefix to a saved figure, which will then include 
    additional info on which column is rendered. 
    :param str mousetype: Type of mouse for figure title purpose.
    :param dist: Distribution to render, defaults to standard normal 
    distribution. Check following for more detail:
    https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html
    :param str dist_name: Name of distribution we compare sample against, for figure title.
    Defaults to default, standard normal.
    :param str line: What line to render for the Q-Q plot, defaults to None.
    :param bool fit: Whether to fit the sample distribution to 'dist' before plotting, 
    thereby attempting to make it into a 45 degree line. Defaults to False.
    :param bool save_figure: Whether to save figure, defaults to True.
    :param bool show_figure: Whether to show figure, defaults to True.
    """
    for col in columns:
        if col not in df.columns: 
            print(f"{col} does not seem to be part of df.columns...")
            print(f"df.columns: {df.columns}")
            continue
        
        col_data = df[col]
        _, ax = plt.subplots()
        # render the plot
        if dist is None:
            sm.qqplot(col_data.to_numpy(), line=line, fit=fit, ax=ax)
        else:
            sm.qqplot(col_data.to_numpy(), dist=dist, line=line, fit=fit, ax=ax)
        # set titles and axis labels
        ax.set_title(f"Q-Q Plot of {dist_name} Distribution Against {col} ({mousetype})")
        
        if save_figure:
            plt.savefig(os.path.join(save_dir, f"{save_prefix}_{dist_name}_{col}_{mousetype}.png"))

        if show_figure: plt.show()
        else: plt.close()

if __name__ == "__main__":
    from visualize_individual_timeseries_data import normalize_columns_relative_to_entry, normalize_distanceByIntervals, RENAMED_DISTBYINTER_NORM

    FILENAME = 'fileName'
    MOUSETYPE = 'mouseType'
    TOTAL_DISTANCE_CM = 'totalDistanceCm'
    CENTER_TIME = 'centerTime'
    CENTERTIME_BY_INTERVALS = "centerTimeByIntervals"
    TIME_FRACTION_BY_QUADRANT = 'timeFractionByQuadrant'
    DISTANCE_BY_INTERVALS = 'distanceByIntervals'
    DISTANCE_BY_INTERVALS_NORM = 'distanceByIntervals_norm'

    for mousetype in [
        # "Q175", 
        "YAC128"
        ]:
        
        SAVE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\{}\fig\qq".format(mousetype)
        SAVE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\{}\fig\qq_NoWeirdMice".format(mousetype)

        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)

        CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
        CSV_PATHS = [
            # os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_{}_analysis_data_filt.csv".format(mousetype)),
            # os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_{}_analysis_data_unfilt.csv".format(mousetype))
            # os.path.join(CSV_FOLDER, r"without_349412m5\No349412m5_WithCenterTimeOverTime_YAC128_analysis_data_filt.csv"),
            # os.path.join(CSV_FOLDER, r"without_349412m5\No349412m5_WithCenterTimeOverTime_YAC128_analysis_data_unfilt.csv"),
            os.path.join(CSV_FOLDER, r"without_weird_YACs\NoWeirdMice_WithCenterTimeOverTime_YAC128_analysis_data_filt.csv"),
            os.path.join(CSV_FOLDER, r"without_weird_YACs\NoWeirdMice_WithCenterTimeOverTime_YAC128_analysis_data_unfilt.csv"),
        ]

        renaming_columns = {
            CENTERTIME_BY_INTERVALS : f"{CENTERTIME_BY_INTERVALS} (0~5)",
            'Unnamed: 5' : f"{CENTERTIME_BY_INTERVALS} (5~10)",
            'Unnamed: 6' : f"{CENTERTIME_BY_INTERVALS} (10~15)",
            'Unnamed: 7' : f"{CENTERTIME_BY_INTERVALS} (15~20)",
            'Unnamed: 8' : f"{CENTERTIME_BY_INTERVALS} (20~25)",
            'Unnamed: 9' : f"{CENTERTIME_BY_INTERVALS} (25~30)",
            DISTANCE_BY_INTERVALS_NORM : f"{DISTANCE_BY_INTERVALS} Normalized (0~5)", 
            'Unnamed: 15_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (5~10)",
            'Unnamed: 16_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (10~15)",
            'Unnamed: 17_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (15~20)",
            'Unnamed: 18_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (20~25)",
            'Unnamed: 19_norm' : f"{DISTANCE_BY_INTERVALS} Normalized (25~30)",
            DISTANCE_BY_INTERVALS : f"{DISTANCE_BY_INTERVALS} (0~5)", 
            'Unnamed: 15' : f"{DISTANCE_BY_INTERVALS} (5~10)",
            'Unnamed: 16' : f"{DISTANCE_BY_INTERVALS} (10~15)",
            'Unnamed: 17' : f"{DISTANCE_BY_INTERVALS} (15~20)",
            'Unnamed: 18' : f"{DISTANCE_BY_INTERVALS} (20~25)",
            'Unnamed: 19' : f"{DISTANCE_BY_INTERVALS} (25~30)",
            }

        to_render_orig = [
            # those rendered in scatter plots
            TOTAL_DISTANCE_CM, CENTER_TIME,
            # those rendered in plot over time
            CENTERTIME_BY_INTERVALS, 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
            DISTANCE_BY_INTERVALS, 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19',
            # DISTANCE_BY_INTERVALS_NORM, 'Unnamed: 15_norm', 'Unnamed: 16_norm', 'Unnamed: 17_norm', 'Unnamed: 18_norm', 'Unnamed: 19_norm'
            ] + \
                list(RENAMED_DISTBYINTER_NORM.values())
        
        for csv in CSV_PATHS:
            print(f"Processing csv: {os.path.basename(csv)}!")
            
            filt_or_unfilt = "filt" if ("unfilt" not in csv) else "unfilt"
            
            df = pd.read_csv(csv)
            # produce the normalized distance-by-intervals columns
            norm_df = normalize_distanceByIntervals(df)

            # rename the columns in norm_df
            new_columns = [renaming_columns.get(col, col) for col in norm_df.columns.tolist()]
            norm_df.columns = new_columns

            to_render_new = [renaming_columns.get(entry, entry) for entry in to_render_orig]

            visualize_qq_plot_from_dataframe(
                df=norm_df, columns=to_render_new, 
                save_dir=SAVE_DIR,
                save_prefix=f"QQplot_{filt_or_unfilt}",
                mousetype=mousetype,
                line='s', fit=False,
                save_figure=True, show_figure=False
                )