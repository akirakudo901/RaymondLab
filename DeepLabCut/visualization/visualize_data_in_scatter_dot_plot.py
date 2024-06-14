# Author: Akira Kudo
# Created: 2024/04/26
# Last Updated: 2024/06/14

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEDIAN_LINESTYPE = '-'
MEAN_LINESTYPE = '--'

FILENAME = "fileName"
MOUSETYPE = 'mouseType'
TOTAL_DISTANCE_CM = 'totalDistanceCm'
CENTER_TIME = 'centerTime'
TIME_FRACTION_BY_QUADRANT = 'timeFractionByQuadrant'
DISTANCE_BY_INTERVALS = 'distanceByIntervals'

def visualize_data_in_scatter_dot_plot_from_csv(csv_path : str,
                                                ylabel : str,
                                                y_val : str,
                                                colors : list,
                                                save_dir : str,
                                                save_name : str,
                                                sex_marker : list=['o', 'o'],
                                                x_val : str=MOUSETYPE,
                                                xlabel : str="Mouse Type",
                                                title : str=None,
                                                show_median : bool=False,
                                                show_mean : bool=False,
                                                save_figure : bool=True,
                                                show_figure : bool=True, 
                                                side_by_side : bool=False):
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
    :param list sex_marker: Marker of ['male', 'female'] mice in the dataset to be 
    rendered on the scatter plot. A list of MarkerStyle, as shown [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html). 
    Defaults to ['o', 'o'].
    :param str x_val: Specifies what is rendered as x-label - has to be one
    of columns in the DLC csv. Defaults to MOUSETYPE.
    :param str xlabel: Label for x-axis, defaults to "Mouse Type".
    :param str title: Figure title, defaults to '{xlabel} per {ylabel}'.
    :param bool show_median: Whether to show median on graph, defaults to False.
    :param bool show_mean: Whether to show mean on graph, defaults to False.
    :param bool save_figure: Whether to save figure, defaults to True.
    :param bool show_figure: Whether to show figure, defaults to True.
    :param bool side_by_side: Whether to put the data for male & female on the same 
    plot, or side by side. Defaults to putting it on the same plot.
    """
    df = pd.read_csv(csv_path)
    visualize_data_in_scatter_dot_plot_from_dataframe(
        df=df, ylabel=ylabel, y_val=y_val, colors=colors, save_dir=save_dir, 
        save_name=save_name, sex_marker=sex_marker, x_val=x_val, xlabel=xlabel,
        title=title, show_median=show_median, show_mean=show_mean, 
        save_figure=save_figure, show_figure=show_figure, side_by_side=side_by_side
        )

def visualize_data_in_scatter_dot_plot_from_dataframe(df : pd.DataFrame,
                                                    ylabel : str,
                                                    y_val : str,
                                                    colors : list,
                                                    save_dir : str,
                                                    save_name : str,
                                                    sex_marker : list=['o', 'o'],
                                                    x_val : str=MOUSETYPE,
                                                    xlabel : str="Mouse Type",
                                                    title : str=None,
                                                    show_median : bool=False,
                                                    show_mean : bool=False,
                                                    save_figure : bool=True,
                                                    show_figure : bool=True,
                                                    side_by_side : bool=False):
    """
    Takes in a data frame holding information obtained via running analysis.m
    onto multiple DLC csvs; and holding basic analysis info of open field
    mouse activity, such as mouseType, totalDistanceCm, centerTime etc.

    Plots the specified feature, aggregated by mouseType into a scatter plot.

    :param pd.DataFrame df: Data frame holding DLC data.
    :param str ylabel: Label for y-axis.
    :param str y_val: Specifies what is rendered as y-label - has to be one 
    of columns in the DLC csv.
    :param list colors: A list of colors matching each x entries. If only one 
    is given, every column is of the same color.
    :param str save_dir: Directory for saving figure.
    :param str save_name: Name of saved figure.
    :param list sex_marker: Marker of ['male', 'female'] mice in the dataset to be 
    rendered on the scatter plot. A list of MarkerStyle, as shown [here](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html). 
    Defaults to ['o', 'o'].
    :param str x_val: Specifies what is rendered as x-label - has to be one
    of columns in the DLC csv. Defaults to MOUSETYPE.
    :param str xlabel: Label for x-axis, defaults to "Mouse Type".
    :param str title: Figure title, defaults to '{xlabel} per {ylabel}'.
    :param bool show_median: Whether to show median on graph, defaults to False.
    :param bool show_mean: Whether to show mean on graph, defaults to False.
    :param bool save_figure: Whether to save figure, defaults to True.
    :param bool show_figure: Whether to show figure, defaults to True.
    :param bool side_by_side: Whether to put the data for male & female on the same 
    plot, or side by side. Defaults to putting it on the same plot.
    """
    sex_marker_to_label = {'male' : sex_marker[0], 'female' : sex_marker[1]}

    unique_mousetypes = np.unique(df[MOUSETYPE])
    # make sure enough colors are specified unambiguously
    if len(colors) != 1 and len(colors) != len(unique_mousetypes):
        raise Exception("Given color have to match the number of mouse types in csv...")
    # make sure there are two markers specified as sex_marker
    if len(sex_marker) != 2:
        raise Exception("Exactly two entries have to be specified for sex_marker, " +
                        f"but we were provided with {len(sex_marker)}...")
    # otherwise if only one color is specified, set color to the same one
    if len(colors) == 1:
        colors = colors * len(unique_mousetypes)
    # set default title if not given
    if title is None:
        title = f'{xlabel} per {ylabel}'
    
    # set the legend manually for median and mean
    median_line = mlines.Line2D([],[], color='black', linestyle=MEDIAN_LINESTYPE)
    mean_line   = mlines.Line2D([],[], color='black', linestyle=MEAN_LINESTYPE)
    handles, labels = [median_line, mean_line], ['Median', 'Mean']

    # create the subplots with different configs for side-by-side or not
    ncols = 2 if side_by_side else 1
    figsize = (5, 6) if side_by_side else (3,6)
    _, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
    
    for mstp_idx, mstp in enumerate(unique_mousetypes):
        X = df[x_val].loc[df[MOUSETYPE] == mstp]
        Y = df[y_val].loc[df[MOUSETYPE] == mstp]
            
        for sex_idx, (sex, marker) in enumerate(sex_marker_to_label.items()):
            ax = axes[sex_idx] if side_by_side else axes

            sex_abbrev = sex[0] #use the first letter as abbreviation
            X_by_sex = X.loc[df[FILENAME].str.contains(sex_abbrev)]
            Y_by_sex = Y.loc[df[FILENAME].str.contains(sex_abbrev)]

            ax.scatter(X_by_sex, Y_by_sex, color=colors[mstp_idx], marker=marker)
            
            # set boundaries for mean & median bars
            barwidth = 1/3/(len(unique_mousetypes) + 1)
            xmax = (mstp_idx+1) / (len(unique_mousetypes) + 1) - barwidth
            xmin = (mstp_idx+1) / (len(unique_mousetypes) + 1) + barwidth
            # if show_mean is true, show mean
            if show_mean:
                if side_by_side:
                    ax.axhline(y=np.mean(Y_by_sex), xmax=xmax, xmin=xmin,
                               color=colors[mstp_idx], linestyle=MEAN_LINESTYPE)
                    # also include the handles manually
                    ax.legend(handles=handles, labels=labels)
                elif sex_idx == 0:
                    ax.axhline(y=np.mean(Y), xmax=xmax, xmin=xmin,
                               color=colors[mstp_idx], linestyle=MEAN_LINESTYPE)
            # if show_median is true, show median
            if show_median:
                if side_by_side:
                    ax.axhline(y=np.median(Y_by_sex), xmax=xmax, xmin=xmin,
                               color=colors[mstp_idx], linestyle=MEDIAN_LINESTYPE)
                    # also include the handles manually
                    ax.legend(handles=handles, labels=labels)
                elif sex_idx == 0:
                    ax.axhline(y=np.median(Y), xmax=xmax, xmin=xmin,
                               color=colors[mstp_idx], linestyle=MEDIAN_LINESTYPE)
            
            # if side by side, set axis labels
            if side_by_side:
                all_Y = df[y_val]
                ylim_margin = np.ptp(all_Y)/6
                ax.set_ylim(top=np.max(all_Y) + ylim_margin, 
                            bottom=np.min(all_Y) - ylim_margin)
                ax.set_xlabel(f"{xlabel} ({sex})")
                ax.set_xticks(range(-1, len(unique_mousetypes)+1))
                if sex_idx == 0:
                    ax.set_ylabel(ylabel)
    
    # if not side by side, add legend to the single figure
    if not side_by_side:
        # add sex markers if they differ
        if sex_marker[0] != sex_marker[1]:
            sex_handles = [mlines.Line2D([],[], color='black', marker=marker, linestyle='None')
                        for marker in sex_marker_to_label.values()]
            sex_labels = list(sex_marker_to_label.keys())
            handles, labels = handles + sex_handles, labels + sex_labels
        plt.legend(handles=handles, labels=labels)

        # set the other settings
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.xticks(range(-1, len(unique_mousetypes)+1), rotation=45)
    
    plt.suptitle(title)
    plt.tight_layout()

    if save_figure:
        plt.savefig(os.path.join(save_dir, save_name))
    
    if show_figure:
        plt.show()
    else:
        plt.close()

    
if __name__ == "__main__":
    SAVE_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\COMPUTED"
    
    CSV_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\previous\openfield\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
    CSV_NAME = "Q175_analysis_data_filt.csv"
    CSVFILE = os.path.join(CSV_DIR, CSV_NAME)

    MOUSETYPE = "Q175"

    filter_or_unfiltered = CSV_NAME.replace('.csv', '').split('_')[-1]

    to_visualize = [
        [TOTAL_DISTANCE_CM, 'Total Distance (cm)', f"TotalDistanceCmPerMouseType_{MOUSETYPE}_{filter_or_unfiltered}.png"],
        [CENTER_TIME,       'Center Time (%)',     f"CenterTimePerMouseType_{MOUSETYPE}_{filter_or_unfiltered}.png"]
    ]

    # TIME_FRACTION_BY_QUADRANT : 'Time Fraction By Quadrant (%)', 
    # DISTANCE_BY_INTERVALS : 'Distance By Intervals (cm)'
    
    for visualized_var, var_name, fig_name in to_visualize:
        visualize_data_in_scatter_dot_plot_from_csv(CSVFILE,
                                            y_val=visualized_var,
                                            xlabel=f'Mouse Type ({MOUSETYPE})',
                                            ylabel=var_name,
                                            title=f"{var_name} Per Mouse Type",
                                            colors=['red', 'blue'],
                                            save_figure=True,
                                            save_dir=SAVE_DIR,
                                            save_name=fig_name,
                                            show_figure=False,
                                            show_mean=True,
                                            show_median=True)