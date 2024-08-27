# Author: Akira Kudo
# Created: 2024/04/28
# Last Updated: 2024/08/06

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import yaml

from visualization.visualize_data_in_dot_and_whisker_plot import visualize_data_in_dot_and_whisker_plot_from_dataframe, visualize_data_per_individual_in_quadrants, FILENAME, CENTERTIME_BY_INTERVALS, TIME_FRACTION_BY_QUADRANT, DISTANCE_BY_INTERVALS
from visualization.visualize_data_in_scatter_dot_plot import visualize_data_in_scatter_dot_plot_from_dataframe, CENTER_TIME, FILENAME, MOUSETYPE, TOTAL_DISTANCE_CM
from visualization.visualize_individual_timeseries_data import visualize_individual_timeseries_data_from_csv, visualize_individual_timeseries_data_from_dataframe, normalize_distanceByIntervals, RENAMED_DISTBYINTER_NORM

def do_visualize_dlc_basic_analysis_in_plot(csvfile : str,
                                            save_dir : str,
                                            mouse_groupname : str,
                                            filtered : bool,
                                            truncated : bool, 
                                            colors : list, 
                                            save_figure : bool=True, 
                                            show_figure : bool=True,
                                            sex_side_by_side : bool=False):
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
    
    # visualize while putting male and female together and separate
    male_df   = norm_df[norm_df[FILENAME].str.contains('m')]
    female_df = norm_df[norm_df[FILENAME].str.contains('f')]

    # save figures in different folders: mNf, m and f
    all_df = [norm_df]; sex_folders = ['mNf']; sex_name = ['Male And Female']
    # we will only create the male only & female if we aren't creating side by side 
    # plots in terms of the different sexes
    if not sex_side_by_side and len(male_df) != 0: 
        all_df.append(male_df); sex_folders.append('m'); sex_name.append('Male Only')
    if not sex_side_by_side and len(female_df) != 0: 
        all_df.append(female_df); sex_folders.append('f'); sex_name.append('Female Only')

    for sex_diff_df, foldername, sex in zip(all_df, sex_folders, sex_name):
        fullpath_sexfolder = os.path.join(save_dir, foldername)
        if not os.path.exists(fullpath_sexfolder):
            os.mkdir(fullpath_sexfolder)
        
        if True:
            for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_with_dot_and_whisker:
                visualize_data_in_dot_and_whisker_plot_from_dataframe(
                    df=sex_diff_df,
                    x_vals=x_vals,
                    y_vals=visualized_var,
                    xlabel=xlabel,
                    ylabel=var_name,
                    title=f"{var_name} Per Mouse Type ({mouse_groupname}, {sex})",
                    colors=colors,
                    save_figure=save_figure,
                    save_dir=fullpath_sexfolder,
                    save_name=fig_name,
                    show_figure=show_figure
                    )
        
        if True:
            for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_with_individuals_in_quadrants:
                visualize_data_per_individual_in_quadrants(csvfile,
                                                    x_vals=x_vals,
                                                    y_vals=visualized_var,
                                                    xlabel=xlabel,
                                                    ylabel=var_name,
                                                    title=f"{var_name} Per Mouse Type ({mouse_groupname})",
                                                    colors=colors,
                                                    save_figure=save_figure,
                                                    save_dir=fullpath_sexfolder,
                                                    save_name=fig_name,
                                                    show_figure=show_figure, 
                                                    vmin=0, vmax=100) # quadrant by percentage!
        
        if True:
            for visualized_var, var_name, fig_name in to_visualize_scatter:
                visualize_data_in_scatter_dot_plot_from_dataframe(sex_diff_df,
                                                    y_val=visualized_var,
                                                    xlabel=f'Mouse Type',
                                                    ylabel=var_name,
                                                    title=f"{var_name} Per Mouse Type ({mouse_groupname}, {sex})",
                                                    colors=colors,
                                                    # sex_marker=['.', 'x'],
                                                    sex_marker=['.', '.'],
                                                    save_figure=save_figure,
                                                    save_dir=fullpath_sexfolder,
                                                    save_name=fig_name,
                                                    # show_figure=show_figure, TODO FIX
                                                    show_figure=True, #TODO FIX
                                                    show_mean=True,
                                                    show_median=True,
                                                    side_by_side=sex_side_by_side)
        
        if True:
            for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_individuals_as_timeseries:
                visualize_individual_timeseries_data_from_dataframe(
                    df=sex_diff_df,
                    x_vals=x_vals,
                    y_vals=visualized_var,
                    xlabel=xlabel,
                    ylabel=var_name,
                    title=f"{var_name} Per Mouse Type ({mouse_groupname}, {sex})",
                    colors=colors,
                    save_figure=save_figure,
                    save_dir=fullpath_sexfolder,
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

    if False:
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
                                                    save_figure=False,
                                                    show_figure=False,
                                                    sex_side_by_side=True)
            
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
        WHICH_MICE = "AllMice" if False else "No535m1_153m2"
        
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
    
    # compare the weights of YACs and WTs
    if True:
        WEIGHT_YAML = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\YAC128_Weights_from_RaymondLab.yaml"

        FIGURE_SAVING_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\fig\pawInk\fig_workTermReport"
        
        WT, HD = "WT", "YAC128"
        WEIGHT = "Weight"
        MOUSENAME = "MouseName"
        # significance with Mann Whitney U
        SIGNIFICANCE = 0.05
        SAVE_COMPARISON_RESULT = True
        
        with open(WEIGHT_YAML, 'r') as f:
            yaml_content = f.read()
        content = yaml.safe_load(yaml_content)

        wt_weights = list(content['WT'].values())
        hd_weights = list(content['YAC128'].values())
        
        all_weights = wt_weights + hd_weights
        mousetypes = [WT] * len(wt_weights) + [HD] * len(hd_weights)

        df = pd.DataFrame(data={FILENAME : list(content['WT'].keys()) + list(content['YAC128'].keys()), 
                                WEIGHT : all_weights, 
                                MOUSETYPE : mousetypes})
        
        if False:
            visualize_data_in_scatter_dot_plot_from_dataframe(
                df=df,
                y_val=WEIGHT,
                xlabel='Mouse Type',
                ylabel="Weight",
                title=f"Weight Per Mouse Type (YAC128, Male)",
                colors=["black", "pink"],
                # sex_marker=['.', 'x'],
                sex_marker=['.', '.'],
                save_figure=True,
                save_dir=FIGURE_SAVING_PATH,
                save_name=f"YAC128_Subset_Weight_ScatterPlot.png",
                show_figure=False,
                show_mean=True,
                show_median=True,
                side_by_side=False
                )
        
        # do a statistical analysis using the Mann Whitney U
        if True:
            _, p_val = mannwhitneyu(wt_weights, hd_weights)
            
            result_text = f"MWU test on {WEIGHT}:\n" + \
                f"- {WT} (n={len(wt_weights)}, mean={np.mean(wt_weights)}),\n" + \
                f"  {HD} (n={len(hd_weights)}, mean={np.mean(hd_weights)}))\n" + \
                f"- p={p_val}; Significance level {'not achieved...' if p_val > SIGNIFICANCE else 'achieved!'}"
            print(result_text)

            if SAVE_COMPARISON_RESULT:
                COMPARISON_RESULT_SAVENAME = "YAC128_Weight_Comparison_MannWhitneyU_Result.txt"
                with open(os.path.join(FIGURE_SAVING_PATH, COMPARISON_RESULT_SAVENAME), 'w') as f:
                    f.write(result_text)