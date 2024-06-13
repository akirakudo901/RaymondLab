# Author: Akira Kudo
# Created: 2024/04/28
# Last Updated: 2024/06/13

import os

import numpy as np
import pandas as pd

from visualization.visualize_data_in_dot_and_whisker_plot import visualize_data_in_dot_and_whisker_plot_from_dataframe, visualize_data_per_individual_in_quadrants, CENTERTIME_BY_INTERVALS, TIME_FRACTION_BY_QUADRANT, DISTANCE_BY_INTERVALS
from visualization.visualize_data_in_scatter_dot_plot import visualize_data_in_scatter_dot_plot, CENTER_TIME, TOTAL_DISTANCE_CM
from visualization.visualize_individual_timeseries_data import visualize_individual_timeseries_data_from_csv, visualize_individual_timeseries_data_from_dataframe, normalize_distanceByIntervals, RENAMED_DISTBYINTER_NORM

def do_visualize_dlc_basic_analysis_in_plot(csvfile : str,
                                            save_dir : str,
                                            mouse_groupname : str,
                                            filtered : bool,
                                            truncated : bool, 
                                            colors : list, 
                                            save_figure : bool=True, 
                                            show_figure : bool=True):
    # first determine the rendering file name
    filtered_or_unfiltered = 'filtered' if filtered else 'unfiltered'
    trunc_or_untrunc = 'trunc' if truncated else 'untrunc'
    suffix = f"{mouse_groupname}_{filtered_or_unfiltered}_{trunc_or_untrunc}"

    # define the details of the visualization for dot & whisker plot
    to_visualize_with_dot_and_whisker = [
        [[DISTANCE_BY_INTERVALS, 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19'],
         ['0~5','5~10','10~15','15~20','20~25','25~30'],
         "Interval (min)",
         'Distance By Intervals (cm)',     
         f"DistanceByIntervalsCmPerMouseType_{suffix}.png"],
         [[CENTERTIME_BY_INTERVALS, 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9'],
         ['0~5','5~10','10~15','15~20','20~25','25~30'],
         "Interval (min)",
         'Center Time By Intervals (%)',     
         f"CenterTimeByIntervalsPercPerMouseType_{suffix}.png"],
         [list(RENAMED_DISTBYINTER_NORM.values()),
        ['0~5','5~10','10~15','15~20','20~25','25~30'],
        "Interval (min)",
        'Distance By Intervals Normalized To First Interval (ratio)',     
        f"DistanceByIntervalsNormPerMouseType_{suffix}.png"]
    ]

    # define the details of the visualization for individuals per quadrants
    to_visualize_with_individuals_in_quadrants = [
        [[TIME_FRACTION_BY_QUADRANT, 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'], 
         ['1', '2', '3', '4'],
         "Quadrant number",
         'Time Fraction By Quadrant (%)', 
         f"TimeFractionByQuadrantPerMouseType_{suffix}.png"]
    ]

    # define the details of the visualization for scatter plot
    to_visualize_scatter = [
        [TOTAL_DISTANCE_CM, 'Total Distance (cm)', f"TotalDistanceCmPerMouseType_{suffix}.png"],
        [CENTER_TIME,       'Center Time (%)',     f"CenterTimePerMouseType_{suffix}.png"]
    ]

    # define the details of the visualization for individuals over time
    to_visualize_individuals_as_timeseries = [
        [[DISTANCE_BY_INTERVALS, 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19'],
        ['0~5','5~10','10~15','15~20','20~25','25~30'],
        "Interval (min)",
        'Distance By Intervals (cm)',     
        f"DistanceByIntervalsCmPerIndividualMouse_{suffix}.png"],
        [[CENTERTIME_BY_INTERVALS, 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9'],
        ['0~5','5~10','10~15','15~20','20~25','25~30'],
        "Interval (min)",
        'Center Time By Intervals (%)',
        f"CenterTimeByIntervalsPercPerIndividualMouse_{suffix}.png"],
        [list(RENAMED_DISTBYINTER_NORM.values()),
        ['0~5','5~10','10~15','15~20','20~25','25~30'],
        "Interval (min)",
        'Distance By Intervals Normalized To First Interval (ratio)',     
        f"DistanceByIntervalsNormPerIndividualMouse_{suffix}.png"]
        ]
    
    # create the normalized entries for the data frame
    df = pd.read_csv(csvfile)
    if not np.all(np.isin(list(RENAMED_DISTBYINTER_NORM.values()), df.columns)):
        norm_df = normalize_distanceByIntervals(df)
    else:
        norm_df = df
    
    # visualize
    for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_with_dot_and_whisker:
        visualize_data_in_dot_and_whisker_plot_from_dataframe(
            df=norm_df,
            x_vals=x_vals,
            y_vals=visualized_var,
            xlabel=xlabel,
            ylabel=var_name,
            title=f"{var_name} Per Mouse Type ({mouse_groupname})",
            colors=colors,
            save_figure=save_figure,
            save_dir=save_dir,
            save_name=fig_name,
            show_figure=show_figure
            )
    
    for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_with_individuals_in_quadrants:
        visualize_data_per_individual_in_quadrants(csvfile,
                                            x_vals=x_vals,
                                            y_vals=visualized_var,
                                            xlabel=xlabel,
                                            ylabel=var_name,
                                            title=f"{var_name} Per Mouse Type ({mouse_groupname})",
                                            colors=colors,
                                            save_figure=save_figure,
                                            save_dir=save_dir,
                                            save_name=fig_name,
                                            show_figure=show_figure, 
                                            vmin=0, vmax=100) # quadrant by percentage!

    for visualized_var, var_name, fig_name in to_visualize_scatter:
        visualize_data_in_scatter_dot_plot(csvfile,
                                            y_val=visualized_var,
                                            xlabel=f'Mouse Type ({mouse_groupname})',
                                            ylabel=var_name,
                                            title=f"{var_name} Per Mouse Type",
                                            colors=colors,
                                            sex_marker=['.', 'x'],
                                            save_figure=save_figure,
                                            save_dir=save_dir,
                                            save_name=fig_name,
                                            show_figure=show_figure,
                                            show_mean=True,
                                            show_median=True)
    
    for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_individuals_as_timeseries:
        visualize_individual_timeseries_data_from_dataframe(
            df=norm_df,
            x_vals=x_vals,
            y_vals=visualized_var,
            xlabel=xlabel,
            ylabel=var_name,
            title=f"{var_name} Per Mouse Type ({mouse_groupname})",
            colors=colors,
            save_figure=save_figure,
            save_dir=save_dir,
            save_name=fig_name,
            show_figure=show_figure)

if __name__ == "__main__":
    def render_figures_for_group(mouse_groupname):
        ########################################
        # NOTHING ELSE HAS TO (ARGUABLY) CHANGE!
        ########################################
        SAVE_DIR = r"C:\Users\mashi\Desktop\temp\{}\basic_analysis\fig".format(mouse_groupname)        

        CSV_DIR = r"C:\Users\mashi\Desktop\temp\{}\basic_analysis".format(mouse_groupname)
        CSV_NAMES = [
            f"{mouse_groupname}_analysis_data_trunc_unfilt.csv",
            f"{mouse_groupname}_analysis_data_trunc_filt.csv"
        ]

        colors = ['black', 'pink'] if mouse_groupname == "YAC128" else \
                (['red', 'blue'] if mouse_groupname == "Q175" else ['green', 'purple'])

        for csv_name in CSV_NAMES:
            csvfile = os.path.join(CSV_DIR, csv_name)
            filter_or_unfiltered = (csv_name.replace('.csv', '').split('_')[-1]) == 'filt'
            trunc_or_untrunc = (csv_name.replace('.csv', '').split('_')[-2]) == 'trunc'

            do_visualize_dlc_basic_analysis_in_plot(csvfile=csvfile,
                                                    save_dir=SAVE_DIR,
                                                    mouse_groupname=mouse_groupname,
                                                    filtered=filter_or_unfiltered,
                                                    truncated=trunc_or_untrunc, 
                                                    colors=colors, 
                                                    save_figure=True,
                                                    show_figure=False)
    
    # render_figures_for_group("Q175")
    # render_figures_for_group("YAC128")

    if True:
        MOUSE_GROUPNAME = "Q175"
        SAVE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\{}\fig\basicAnalysis{}".format(
            MOUSE_GROUPNAME, "AllMice")
        
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)

        CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
        CSV_PATHS = [
            os.path.join(CSV_FOLDER, r"AllMice\Q175_BrownNBlack_analysis_data_filt.csv"),
            os.path.join(CSV_FOLDER, r"AllMice\Q175_BrownNBlack_analysis_data_unfilt.csv"),
        ]

        colors = ['red', 'blue']

        for csv in CSV_PATHS:
            do_visualize_dlc_basic_analysis_in_plot(csvfile=csv,
                                                    save_dir=SAVE_DIR,
                                                    mouse_groupname=MOUSE_GROUPNAME,
                                                    filtered='unfilt' not in csv,
                                                    truncated=False, 
                                                    colors=colors,
                                                    save_figure=True,
                                                    show_figure=False)
            
    if False:
        MOUSE_GROUPNAME = "Q175Black"
        SAVE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\Q175\fig\RangeCorrected"

        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        
        CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
        CSV_PATHS = [
            os.path.join(CSV_FOLDER, "TrialForPrespecifiedRanges_Q175Black_analysis_data_filt.csv"),
            os.path.join(CSV_FOLDER, "TrialForPrespecifiedRanges_Q175Black_analysis_data_unfilt.csv"),

            # os.path.join(CSV_FOLDER, "NoRangeCorrection_Q175Black_analysis_data_filt.csv"),
            # os.path.join(CSV_FOLDER, "NoRangeCorrection_Q175Black_analysis_data_unfilt.csv"),
        ]

        colors = ['red', 'blue']

        for csv in CSV_PATHS:
            do_visualize_dlc_basic_analysis_in_plot(csvfile=csv,
                                                    save_dir=SAVE_DIR,
                                                    mouse_groupname=MOUSE_GROUPNAME,
                                                    filtered='unfilt' not in csv,
                                                    truncated=False, 
                                                    colors=colors,
                                                    save_figure=True,
                                                    show_figure=False)
    
    if False:
        MOUSE_GROUPNAME = "YAC128"
        WHICH_MICE = "AllMice" if True else "No535m1_153m2"
        
        SAVE_DIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\{}\fig\basicAnalysis{}".format(
            MOUSE_GROUPNAME, WHICH_MICE)
        
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        
        CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
        if WHICH_MICE == "AllMice":
            CSV_PATHS = [
                os.path.join(CSV_FOLDER, r"AllMice\YAC128_analysis_data_filt.csv"),
                os.path.join(CSV_FOLDER, r"AllMice\YAC128_analysis_data_unfilt.csv"), 
            ]
        elif WHICH_MICE == "No535m1_153m2":
            CSV_PATHS = [
                os.path.join(CSV_FOLDER, r"OutlierRemoved\NoOutlier_YAC128_analysis_data_filt.csv"),
                os.path.join(CSV_FOLDER, r"OutlierRemoved\NoOutlier_YAC128_analysis_data_unfilt.csv"), 
            ]

        colors = ['black', 'pink']

        for csv in CSV_PATHS:
            do_visualize_dlc_basic_analysis_in_plot(csvfile=csv,
                                                    save_dir=SAVE_DIR,
                                                    mouse_groupname=MOUSE_GROUPNAME,
                                                    filtered='unfilt' not in csv,
                                                    truncated=False,
                                                    colors=colors,
                                                    save_figure=True,
                                                    show_figure=False)