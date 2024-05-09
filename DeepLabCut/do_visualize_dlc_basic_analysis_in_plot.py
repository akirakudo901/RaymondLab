# Author: Akira Kudo
# Created: 2024/04/28
# Last Updated: 2024/04/29

import os

from visualization.visualize_data_in_dot_and_whisker_plot import visualize_data_in_dot_and_whisker_plot, visualize_data_per_individual_in_quadrants, TIME_FRACTION_BY_QUADRANT, DISTANCE_BY_INTERVALS
from visualization.visualize_data_in_scatter_dot_plot import visualize_data_in_scatter_dot_plot, CENTER_TIME, TOTAL_DISTANCE_CM

def do_visualize_dlc_basic_analysis_in_plot(csvfile : str,
                                            save_dir : str,
                                            mouse_groupname : str,
                                            filtered : bool,
                                            truncated : bool, 
                                            colors : list):
    # first determine the rendering file name
    filtered_or_unfiltered = 'filtered' if filtered else 'unfiltered'
    trunc_or_untrunc = 'trunc' if truncated else 'untrunc'
    suffix = f"{mouse_groupname}_{filtered_or_unfiltered}_{trunc_or_untrunc}"

    # define the details of the visualization for dot & whisker plot
    to_visualize_with_dot_and_whisker = [
        [[DISTANCE_BY_INTERVALS, 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13'],
         ['0~5','5~10','10~15','15~20','20~25','25~30'],
         "Interval (min)",
         'Distance By Intervals (cm)',     
         f"DistanceByIntervalsCmPerMouseType_{suffix}.png"]
    ]

    # define the details of the visualization for individuals per quadrants
    to_visualize_with_individuals_in_quadrants = [
        [[TIME_FRACTION_BY_QUADRANT, 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], 
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

    # visualize
    for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_with_dot_and_whisker:
        visualize_data_in_dot_and_whisker_plot(csvfile,
                                            x_vals=x_vals,
                                            y_vals=visualized_var,
                                            xlabel=xlabel,
                                            ylabel=var_name,
                                            title=f"{var_name} Per Mouse Type",
                                            colors=colors,
                                            save_figure=True,
                                            save_dir=save_dir,
                                            save_name=fig_name,
                                            show_figure=False)
    
    for visualized_var, x_vals, xlabel, var_name, fig_name in to_visualize_with_individuals_in_quadrants:
        visualize_data_per_individual_in_quadrants(csvfile,
                                            x_vals=x_vals,
                                            y_vals=visualized_var,
                                            xlabel=xlabel,
                                            ylabel=var_name,
                                            title=f"{var_name} Per Mouse Type ({mouse_groupname})",
                                            colors=colors,
                                            save_figure=True,
                                            save_dir=save_dir,
                                            save_name=fig_name,
                                            show_figure=False, 
                                            vmin=0, vmax=100) # quadrant by percentage!

    for visualized_var, var_name, fig_name in to_visualize_scatter:
        visualize_data_in_scatter_dot_plot(csvfile,
                                            y_val=visualized_var,
                                            xlabel=f'Mouse Type ({mouse_groupname})',
                                            ylabel=var_name,
                                            title=f"{var_name} Per Mouse Type",
                                            colors=colors,
                                            save_figure=True,
                                            save_dir=save_dir,
                                            save_name=fig_name,
                                            show_figure=False,
                                            show_mean=True,
                                            show_median=True)

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
                                                    colors=colors)
    
    render_figures_for_group("Q175")
    render_figures_for_group("YAC128")