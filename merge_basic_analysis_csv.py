# Author: Akira Kudo
# Created: 2024/06/12
# Last Updated: 2024/06/12

import pandas as pd
from DeepLabCut.visualization.visualize_individual_timeseries_data import normalize_distanceByIntervals

COLUMNS = ["fileName", "mouseType", "totalDistanceCm", "centerTime",
           "centerTimeByIntervals", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9",
           "timeFractionByQuadrant", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13",
           "distanceByIntervals", "Unnamed: 15", "Unnamed: 16", "Unnamed: 17", "Unnamed: 18", "Unnamed: 19",
           "distanceByIntervals Normalized (0~5)", "distanceByIntervals Normalized (5~10)",
           "distanceByIntervals Normalized (10~15)", "distanceByIntervals Normalized (15~20)",
           "distanceByIntervals Normalized (20~25)", "distanceByIntervals Normalized (25~30)"
           ]

def merge_basic_analysis_csv(csvlist : list):
    merged_df = pd.DataFrame([], columns=COLUMNS)
    for csv in csvlist:
        df = pd.read_csv(csv)
        if "Unnamed: 0" in df.columns:
            df = pd.read_csv(csv, index_col=[0])
        
        if df.columns.tolist() != COLUMNS:
            df = normalize_distanceByIntervals(df, 
                                               os.path.dirname(csv), 
                                               os.path.basename(csv))
        merged_df = pd.concat([merged_df, df])
        merged_df.reset_index(drop=True, inplace=True)
    
    return merged_df



if __name__ == "__main__":
    import os

    ABOVE_CSV = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\OpenField\3part1 MatlabAndPrismAnalysis\MATLAB\openfield_photometry_30min_DLC\data\results"
    
    LIST_OF_CSVLIST = [
        [
            os.path.join(ABOVE_CSV, "TrialForPrespecifiedRanges_Q175Black_analysis_data_filt.csv"),
            os.path.join(ABOVE_CSV, "WithCenterTimeOverTime_Q175_analysis_data_filt.csv"),
        ],
        [
            os.path.join(ABOVE_CSV, "TrialForPrespecifiedRanges_Q175Black_analysis_data_unfilt.csv"),
            os.path.join(ABOVE_CSV, "WithCenterTimeOverTime_Q175_analysis_data_unfilt.csv"),
        ]
    ]

    SAVE_DIR = ABOVE_CSV
    
    for csvlist in LIST_OF_CSVLIST:
        merged_df = merge_basic_analysis_csv(csvlist=csvlist)
        print(merged_df)
        savepath = os.path.join(SAVE_DIR, 
                                f"Q175_BrownNBlack_analysis_data_{'unfilt' if 'unfilt' in csvlist[0] else 'filt'}.csv")
        merged_df.to_csv(savepath, index=False)