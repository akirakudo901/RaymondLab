# Author: Akira Kudo
# Created: 2024/05/23
# Last Updated: 2024/05/24

from itertools import combinations

import Levenshtein
import numpy as np
import pandas as pd

"""
This script will take a csv holding data of computed Means data from Rotarod 
photometry, and attempts to:
- store all csv result into dictionaries, mousename : trial_nums, where
   - trial_nums = list of trial numbers identified as successfully computed in the csv
- exclude all mice that have all of their trials filled
- for every mouse that remain:
   - identify the list of candidate mice, which trial that exist match nicely
- once we have a pairing of possible mice that are identical except typos, merge them by:
   - deciding which mouse name to keep
   - copying the other mouse's trials into the other

To merge two sets of entry for two mice together:
1) We specify two mice: the main, and the typo.
2) We identify columns for main & typo, revealing which column in each are empty / not empty.
3) For every row in typo:
    - if it isn't empty:
       - check if the row with the same total trial is non-empty in main
          - if it is, copy over a specified set of fields
          - if it isn't skip
    - if it is empty:
       - ignore
"""

FAILED = "Failed_"

def get_mousename_and_trial(filename : str):
    """
    Takes a file name of the form:
      CageNumber + MaleOrFemale + MouseNumber + _ + TrialNumber (+ _ + RepeatNumber)
    and returns:
    - CageNumber + MaleOrFemale + MouseNumber
    - TrialNumber
    - RepeatNumber (None if not there)

    :param str filename: Name of the file we read.
    """
    if filename.startswith(FAILED): 
        prefix, filename = FAILED, filename.replace(FAILED, '')
    split_filename = filename.split('_')
    mousename, trialnumber = split_filename[0], int(split_filename[1])
    if len(split_filename) == 3:
        repeatnumber = int(split_filename[2])
    else:
        repeatnumber = None
    return mousename, trialnumber, repeatnumber

def trials_dont_overlap(trial1 : list, trial2 : list):
    """
    Identifies whether the set of trials don't overlap, returning
    a boolean for whether they match.

    :param list trial1: Trials from mouse 1.
    :param list trial2: Trials from mouse 2.
    """
    for trial in trial1:
        if trial in trial2:
            return False
    return True

def merge_two_mice_data_entries(df : pd.DataFrame,
                                mainmouse : str, 
                                typomouse : str):
    # these are all the fields we copy over when we 'merge' mice data
    COPYOVER_FIELDS = ['Means_Green', 'Meanshift_Green', 'Means_Red', 
                       'Meanshift_Red',
                       'After_TooShort', 'Note_Onset_Size1', 
                       'Note_Onset_Size2', 'PtAB_Onset_Size1', 'PtAB_Onset_Size2', 
                       'Exception', 'Info']
    # we identify rows for main & typo
    mainmouse_rows = df[df[MOUSENAME].astype("string") == str(mainmouse)]
    typomouse_rows = df[df[MOUSENAME].astype("string") == str(typomouse)]
    # for every row in typo
    for _, row in typomouse_rows.iterrows():
        # if it isn't empty, or empty but name start with FAILED
        if not pd.isnull(row[MEANSGREEN]) or row[FILENAME].startswith(FAILED):
            # check if the row with the same total trial is empty in main
            target_row = mainmouse_rows[((mainmouse_rows[TOTALTRIAL] == row[TOTALTRIAL]) &
                                         (~mainmouse_rows[FILENAME].str.startswith(FAILED)))]
            # if it is, copy over a specified set of fields
            if pd.isnull(target_row[MEANSGREEN].item()):
                df.loc[target_row.index, COPYOVER_FIELDS] = row[COPYOVER_FIELDS].to_numpy()
                if row[FILENAME].startswith(FAILED):
                    df.loc[target_row.index, FILENAME] = FAILED + df.loc[target_row.index, FILENAME].item()

    # finally delete entries from typomouse from df
    df.drop(typomouse_rows.index, inplace=True)
    df.reset_index(drop=True, inplace=True)

# we now just have to identify the pairs that are likely to be from the same mouse, then
# we can build a quick csv out of it that stores them as pairs of (main : typo).
# Finally, after every new run to generate data, those pairs can be merged together automatically 
# via using these scripts!
# Now did I save time? Probably not... But you never know, maybe we have to run this over and over!
# It is about reproductivity, my friend!

if __name__ == "__main__":
    CSV_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Rotarod\photometryAnalysis\SignalMeans\signalMeanComputation\results\Q175AdditionalInfo_photometry_means_analysis_result.csv"

    FILENAME = 'fileName'
    MEANSGREEN = 'Means_Green'
    MOUSENAME = 'MouseName'
    TOTALTRIAL = 'TotalTrial'
    EXPECTED_TRIAL_NUM = 12

    if False:
        # first, read the computed value csv
        df = pd.read_csv(CSV_PATH)

        # store all csv result into dictionaries, mousename : trial_nums, where
        #  - trial_nums = list of trial numbers identified as successfully computed in the csv
        mice_trials, matching_mice = {}, {}
        for filename in df[FILENAME]:
            mousename, trialnumber, repeatnumber = get_mousename_and_trial(filename)
            if mousename not in mice_trials.keys(): 
                mice_trials[mousename] = set()
            if pd.isnull(df[df[FILENAME] == filename].loc[:, MEANSGREEN].item()):
                mice_trials[mousename].add(trialnumber)
        # exclude all mice that have all of their trials filled
        mice_trials_keys, mice_trials_values = list(mice_trials.keys()), list(mice_trials.values())
        for mouse, trials in zip(mice_trials_keys, mice_trials_values):
            if len(trials) == EXPECTED_TRIAL_NUM or len(trials) == 0:
                del mice_trials[mouse]
        # for every mouse that remain:
        #  - identify the list of candidate mice, which trial that exist match nicely
        for (mouse, trials), (mouse2, trials2) in combinations(mice_trials.items(), 2):
            if trials_dont_overlap(trials, trials2):
                if len(trials) + len(trials2) > 0:
                    if mouse not in matching_mice.keys(): 
                        matching_mice[mouse] = set()
                    matching_mice[mouse].add(mouse2)
                
        print("matching_mice:")
        for mouse, match in matching_mice.items():
            print(f"mouse: {mouse}")
            match = list(match)
            match.sort(key= lambda x : Levenshtein.distance(mouse, x))
            for item in match:
                if Levenshtein.distance(mouse, item) < 3:
                    print(f"- {item}")
        # - once we have a pairing of possible mice that are identical except typos, merge them by:
        #    - deciding which mouse name to keep
        #    - copying the other mouse's trials into the other
    
    
    if True:
        import os
        import yaml

        YAML_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Rotarod\photometryAnalysis\SignalMeans\signalMeanComputation\results\Q175AdditionalInfo\which_mice_to_merge.yml"
        with open(YAML_PATH, 'r') as file:
            mice_to_merge = yaml.safe_load(file)
            mice_to_merge = mice_to_merge['mice_to_merge']
        
        df = pd.read_csv(CSV_PATH)

        for mouse in mice_to_merge.keys():
            for typomouse in mice_to_merge[mouse]:
                merge_two_mice_data_entries(df, mainmouse=mouse, typomouse=typomouse)
        
        df.to_csv(os.path.join(os.path.dirname(CSV_PATH), 
                               f"SORTED_{os.path.basename(CSV_PATH)}"))