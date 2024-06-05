# Author: Akira Kudo
# Created: 2024/06/03
# Last Updated: 2024/06/04

import numpy as np
from scipy.stats import anderson, normaltest, shapiro

# Based on this page! : 
# https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/

def normality_test_all_three(data : np.ndarray, alpha : float=0.05, show_message : bool=True,
                             do_shap : bool=True, do_dagos : bool=True, do_anderson : bool=True):
    def report_result(test_name : str, stat : float, p : float):
        print(f"{test_name}:")
        print('- Statistics=%.3f, p=%.3f' % (stat, p))
        print(f'-> H0 {"not" if p > alpha else ""} rejected by {test_name}.')
    
    # Shapiro-Wilk Test
    if do_shap:
        shap_stat, shap_p = shapiro(data)
        if show_message:
            report_result("Shapiro-Wilk test", shap_stat, shap_p)
    else:
        shap_stat, shap_p = np.nan, np.nan

    # D'agostino
    if do_dagos:
        if len(data) < 20:
            print(f"D'agostino's test might be invalid given sample size < 20: {len(data)}")
            dagos_stat, dagos_p = np.nan, np.nan
        else:
            dagos_stat, dagos_p = normaltest(data)
            if show_message:
                report_result("D'agostino test", dagos_stat, dagos_p)
    else:
        dagos_stat, dagos_p = np.nan, np.nan
     
    # Anderson-Darling
    if do_anderson:
        result = anderson(data)
        if show_message:
            print('Statistic: %.3f' % result.statistic)
            for i in range(len(result.critical_values)):
                sl, cv = result.significance_level[i], result.critical_values[i]
                if result.statistic < result.critical_values[i]:
                    print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
                else:
                    print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    else:
        result = None
    
    return (shap_stat, shap_p), (dagos_stat, dagos_p), result

if __name__ == "__main__":
    import os
    import pandas as pd
    import sys

    sys.path.append(r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\visualization")
    
    from visualize_individual_timeseries_data import normalize_distanceByIntervals, RENAMED_DISTBYINTER_NORM

    FILENAME = 'fileName'
    MOUSETYPE = 'mouseType'
    TOTAL_DISTANCE_CM = 'totalDistanceCm'
    CENTER_TIME = 'centerTime'
    CENTERTIME_BY_INTERVALS = "centerTimeByIntervals"
    TIME_FRACTION_BY_QUADRANT = 'timeFractionByQuadrant'
    DISTANCE_BY_INTERVALS = 'distanceByIntervals'

    ALPHA = 0.05

    for mousetype in [
        "Q175", 
        # "YAC128"
        ]:
        
        CSV_FOLDER = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
        CSV_PATHS = [
            os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_{}_analysis_data_filt.csv".format(mousetype)),
            os.path.join(CSV_FOLDER, "WithCenterTimeOverTime_{}_analysis_data_unfilt.csv".format(mousetype)),
            # os.path.join(CSV_FOLDER, r"without_349412m5\No349412m5_WithCenterTimeOverTime_YAC128_analysis_data_filt.csv"),
            # os.path.join(CSV_FOLDER, r"without_349412m5\No349412m5_WithCenterTimeOverTime_YAC128_analysis_data_unfilt.csv"),
            os.path.join(CSV_FOLDER, r"without_weird_YACs\NoWeirdMice_WithCenterTimeOverTime_YAC128_analysis_data_filt.csv"),
            os.path.join(CSV_FOLDER, r"without_weird_YACs\NoWeirdMice_WithCenterTimeOverTime_YAC128_analysis_data_unfilt.csv"),
        ]

        # SAVE_DIR = 

        renaming_columns = {
            CENTERTIME_BY_INTERVALS : f"{CENTERTIME_BY_INTERVALS} (0~5)",
            'Unnamed: 5' : f"{CENTERTIME_BY_INTERVALS} (5~10)",
            'Unnamed: 6' : f"{CENTERTIME_BY_INTERVALS} (10~15)",
            'Unnamed: 7' : f"{CENTERTIME_BY_INTERVALS} (15~20)",
            'Unnamed: 8' : f"{CENTERTIME_BY_INTERVALS} (20~25)",
            'Unnamed: 9' : f"{CENTERTIME_BY_INTERVALS} (25~30)",
            DISTANCE_BY_INTERVALS : f"{DISTANCE_BY_INTERVALS} (0~5)", 
            'Unnamed: 15' : f"{DISTANCE_BY_INTERVALS} (5~10)",
            'Unnamed: 16' : f"{DISTANCE_BY_INTERVALS} (10~15)",
            'Unnamed: 17' : f"{DISTANCE_BY_INTERVALS} (15~20)",
            'Unnamed: 18' : f"{DISTANCE_BY_INTERVALS} (20~25)",
            'Unnamed: 19' : f"{DISTANCE_BY_INTERVALS} (25~30)",
            }

        to_render_orig = [
            # those rendered in scatter plots
            # TOTAL_DISTANCE_CM, CENTER_TIME,
            # those rendered in plot over time
            # CENTERTIME_BY_INTERVALS, 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
            # DISTANCE_BY_INTERVALS, 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19'
            ] + \
            list(RENAMED_DISTBYINTER_NORM.values())
        
        for csv in CSV_PATHS:
            print(f"Processing csv: {os.path.basename(csv)}!")

            # store result into a pandas dataframe:
            # columnName | shap_stat | shap_p | dagos_stat | dagos_p | anderson_result (AS MANY COLS AS NEEDED)
            result_columns = ["columnName", "shap_stat", "shap_p", "shap_not_normal", 
                              "dagos_stat", "dagos_p", "dagos_not_normal"]
                              # + anderson_stat + anderson significance levels
            result = []
            
            df = pd.read_csv(csv)
            df = normalize_distanceByIntervals(df)
            print(f"Normalzied df[distanceByIntervals Normalized (15~20)]:")
            print(df[[FILENAME, "distanceByIntervals Normalized (15~20)"]])
            # rename the columns in df
            df.columns = [renaming_columns.get(entry, entry) for entry in df.columns]
            to_render_new = [renaming_columns.get(entry, entry) for entry in to_render_orig]
            
            for col in to_render_new:
                print(f"Column: {col}")
                col_data = df[col]
                (shap_stat, shap_p), (dagos_stat, dagos_p), anderson_result = normality_test_all_three(
                    data=col_data, alpha=ALPHA, show_message=False, 
                    do_shap=True, do_dagos=True, do_anderson=False)
                
                new_row = [col,  shap_stat,  shap_p,  shap_p < ALPHA, 
                                dagos_stat, dagos_p, dagos_p < ALPHA]
                
                if anderson_result is not None:
                    new_row += [anderson_result.statistic] + \
                               [crit_val for crit_val in anderson_result.critical_values] + \
                               [crit_val < anderson_result.statistic 
                                for crit_val in anderson_result.critical_values]
                result.append(new_row)
            
            if anderson_result is not None:
                result_columns += (["anderson_stat"] + 
                                   [f"{sl}_p" for sl in anderson_result.significance_level
                                     if anderson_result is not None] + 
                                   [f"{sl}_not_normal" for sl in anderson_result.significance_level
                                     if anderson_result is not None])

            result_df = pd.DataFrame(data=result, columns=result_columns)
            print(result_df)