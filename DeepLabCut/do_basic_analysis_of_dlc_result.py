# Author: Akira Kudo
# Created: 2024/04/28
# Last Updated: 2024/08/07

import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from visualization.visualize_data_in_scatter_dot_plot import CENTER_TIME, MOUSETYPE, TOTAL_DISTANCE_CM

def MWU_test_on_specific_variable(csv_path : str, 
                                  variable_of_interest : str,
                                  alternative : str='two-sided'):
    """
    Does the Mann-Whitney U test on specific variables as read 
    from the csv that holds basic analysis results.

    :param str csv_path: Path to csv holding analysis info, in the form
    ['mousename', 'mouseType', 'centerTime', ...].
    :param str variable_of_interest: The variable of interest we graph, has
    to be a column name in the csv.
    :param str alternative: Alternative hypothesis of MWU test, 
    {"two-sided", "less", "greater"}.

    :return mwu_test_res.statistic, mwu_test_res.pvalue, mt1_voi, mt2_voi:
    """
    # read csv and get mouse types
    df = pd.read_csv(csv_path)
    mouse_groups = np.unique(df[MOUSETYPE])
    # we expect a two-subject test
    if len(mouse_groups) != 2: raise Exception("Expecting a dataframe with 2 unique mouse types...")
    # extract the variable of interest for each mouse group
    mt1, mt2 = mouse_groups[0], mouse_groups[1]
    mt1_voi = df[variable_of_interest].loc[df[MOUSETYPE] == mt1]
    mt2_voi = df[variable_of_interest].loc[df[MOUSETYPE] == mt2]
    # do the test
    mwu_test_res = mannwhitneyu(mt1_voi, mt2_voi, use_continuity=True, 
                                alternative=alternative)
    return mwu_test_res.statistic, mwu_test_res.pvalue, (mt1, mt1_voi), (mt2, mt2_voi)

def repeated_ANOVA_on_specific_variable(csv_path : str, 
                                        variable_of_interest : str):
    pass

def carry_out_significance_tests_on_csv(csv_path : str,
                                        significance_level : float=0.05,
                                        savedir : str=None,
                                        savename : str=None):
    
    def carry_out_MWU_test_and_print_result(voi : str, 
                                            alternative : str='two-sided'):
        _, p_val, (mt1, tdcm_1), (mt2, tdcm_2) = MWU_test_on_specific_variable(
            csv_path=csv_path,
            variable_of_interest=voi,
            alternative=alternative
            )
        text = f"MWU test on {voi}:\n" + \
               f"- {mt1} (n={len(tdcm_1)}, mean={np.mean(tdcm_1)}),\n" + \
               f"  {mt2} (n={len(tdcm_2)}, mean={np.mean(tdcm_2)}))\n" + \
               f"- p={p_val}; Significance level {'not achieved...' if p_val > significance_level else 'achieved!'}"
        print(text)
        return text

    print(f"Significance level set to: {significance_level}.")
    # first do Mann-Whitney U test on two distributions
    # or if possible, t-test
    # this is for: Total Distance Cm, Center Time
    result1 = carry_out_MWU_test_and_print_result(voi=TOTAL_DISTANCE_CM, alternative='two-sided')
    result2 = carry_out_MWU_test_and_print_result(voi=CENTER_TIME, alternative='two-sided')
    
    # then save the result
    if savedir is not None and savename is not None:
        with open(os.path.join(savedir, savename), 'w') as f:
            f.write(result1 + "\n-----------------\n" + result2)

if __name__ == "__main__":
    CSV_PATHS = [
        # r"C:\Users\mashi\Desktop\temp\YAC128\basic_analysis\YAC128_analysis_data_trunc_filt.csv",
        # r"C:\Users\mashi\Desktop\temp\YAC128\basic_analysis\YAC128_analysis_data_trunc_unfilt.csv",
        # r"C:\Users\mashi\Desktop\temp\Q175\basic_analysis\Q175_analysis_data_trunc_filt.csv",
        # r"C:\Users\mashi\Desktop\temp\Q175\basic_analysis\Q175_analysis_data_trunc_unfilt.csv",
        # r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results\2024_05_28_new_mice_Q175_analysis_data_unfilt.csv"
        # r"C:\Users\mashi\Desktop\temp\YAC128\basic_analysis\YAC128_analysis_data_trunc_unfilt_NONMOVINGMOUSE_DROPPED.csv",
        # r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results\Q175_BrownNBlack_analysis_data_filt.csv"
        r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results\OutlierRemoved\NoOutlier_YAC128_analysis_data_filt.csv"
    ]

    SAVEDIR = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\YAC128\fig\basicAnalysisNo535m1_153m2"

    SIGNIFICANCE_LEVEL = 0.05

    for csv_path in CSV_PATHS:
        carry_out_significance_tests_on_csv(csv_path=csv_path, 
                                            significance_level=SIGNIFICANCE_LEVEL,
                                            savedir=SAVEDIR,
                                            savename="MannWhitneyU_Comparison_BasicAnalysis.txt")