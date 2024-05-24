
import csv
import os
import re

import numpy as np

REFERENCE_PATH = r"./Latency to Fall.csv"

DIFFERENCE_PATH = r"./Rotarod_LatencyToFall.csv"

SAVE_FOLDER = r"."

SAVE_DIFFERENT_NAME_MATCH_FILENAME = r"DifferentNames_Matches_LatencyToFall.csv"
SAVE_DIFFERENT_NAME_MATCH_PATH = os.path.join(SAVE_FOLDER, SAVE_DIFFERENT_NAME_MATCH_FILENAME)

SAVE_SAME_NAME_FILENAME = r"SameName_LatencyToFall.csv"
SAVE_SAME_NAME_PATH = os.path.join(SAVE_FOLDER, SAVE_SAME_NAME_FILENAME)
# how much tolerance in value difference we perceive as match
ERROR_MARGIN = 5
# significant figure rounding to make it easier to see when writing into csv
SIG_FIGS = 2
# whether to include entries with the same name, when looking for matches solely based on 
# the same trial latencies
SAVE_SAME_ENTRYNAME_PAIRS = False

# FUNCTION DEFINITION

def get_standardized_mouse_name(mouseName):
    try:
        maleOrFemale_mouseNumber = re.findall('[mf]\d+', mouseName)[0]
        cageNumber = re.findall('\d+', mouseName)[0]
        return cageNumber + "_" + maleOrFemale_mouseNumber
    except:
        return mouseName

def equivalent_mouse_names(mouseName1, mouseName2):
    try:
        maleOrFemale_mouseNumber1 = re.findall('[mf]\d+', mouseName1)[0]
        cageNumber1 = re.findall('\d+', mouseName1)[0]

        maleOrFemale_mouseNumber2 = re.findall('[mf]\d+', mouseName2)[0]
        cageNumber2 = re.findall('\d+', mouseName2)[0]

        return (maleOrFemale_mouseNumber1 == maleOrFemale_mouseNumber2) and \
            (cageNumber1.endswith(cageNumber2) or cageNumber2.endswith(cageNumber1))
    except:
        return False



# given a reference & difference dict of entries, return a dictionary
# containing each matches found as
# mouseNameInRef, mouseNameInDiff -- 
#                   [list of value in ref, 
#                    list of value in diff, 
#                    difference of two lists]
# error_margin : Values are considered matching if their difference is 
#                smaller than this.
# include_same_name : Whether to include entries that are matched to have the same name
def find_matching_latencyToFall_entry(ref_dict, diff_dict, error_margin : float, include_same_name=True):
    matches = {}
    for diff_mouseName, diff_entries in diff_dict.items():
        for ref_mouseName, ref_entries in ref_dict.items():
            # if both entries have the same name and include_same_name isn't true, skip
            if diff_mouseName == ref_mouseName and not include_same_name: continue
            # if entries from both dict aren't the same length, skip
            elif len(diff_entries) != len(ref_entries): continue
            # otherwise compare elementwise if close
            elif (np.absolute(diff_entries - ref_entries) < error_margin).all():
                matches[ref_mouseName] = [diff_mouseName,
                                          ref_entries.tolist(), 
                                          diff_entries.tolist(), 
                                          np.absolute(diff_entries - ref_entries).tolist()]

    return matches

# given a reference & difference dict of entries, return a dictionary
# containing each filename in reference matched to one filename in difference if 
# an entry that is close in name is found - or an empty entry, if none is found 
# Returned values will of form:
# mouseNameInRef -- [mouseNameInDiff,
#                    list of value in ref, 
#                    list of value in diff, 
#                    difference of two lists]
def match_names_and_their_latencyToFall_entries(ref_dict, diff_dict):
    matches = {}

    # each entry name is of the form: cageNumber[_- ]?[maleOrFemale]mouseNumber
    # we consider two names to be similar if 
    # 1) cageNumber is identical or one cageNumber is a subset of another
    # 2) maleOrFemale is identical
    # 3) mouseNumber is identical

    for diff_mouseName, diff_entries in diff_dict.items():
        for ref_mouseName, ref_entries in ref_dict.items():
            # if two names aren't equivalent, add an empty entry
            if not equivalent_mouse_names(diff_mouseName, ref_mouseName):
                no_match_name = "no_exactname_match_" + diff_mouseName
                if no_match_name not in matches:
                    matches[no_match_name] = ['', 
                                              [0] * len(ref_entries), 
                                              [0] * len(ref_entries),
                                              [0] * len(ref_entries)]
            else:
                # compare elementwise and store result
                matches[ref_mouseName] = [
                    diff_mouseName,
                    ref_entries.tolist(), 
                    diff_entries.tolist(), 
                    np.absolute(diff_entries - ref_entries).tolist()]
                break

    return matches


def get_ref_dict():
    with open(REFERENCE_PATH, mode='r') as ref_csv:
        ref_dict = {}

        ref = csv.reader(ref_csv)
        fileNames = next(ref)
        remaining = list(ref)

        for i, mouseName in enumerate(fileNames):
            if i == 0: continue
            elif mouseName == '': mouseName = f"missingName_{i}"
            # otherwise standardize mouse name
            mouseName = get_standardized_mouse_name(mouseName)

            for lines in remaining:
                if lines[i] == '': break
                latency_in_ith_trial = float(lines[i])
                if mouseName in ref_dict:
                    ref_dict[mouseName].append(latency_in_ith_trial)
                else:
                    ref_dict[mouseName] = [latency_in_ith_trial]
    # convert it all to numpy array for faster comparison
    for key, val in ref_dict.items():
        ref_dict[key] = np.array(val)
    return ref_dict

def get_diff_dict():
    with open(DIFFERENCE_PATH, mode='r') as diff_csv:
        diff_dict = {}

        MOUSENAME_IDX = 1; LATENCY_TO_FALL_IDX = 3

        diff = csv.reader(diff_csv)

        _ = next(diff) #ignore header

        for lines in diff:
            mouseName = get_standardized_mouse_name(lines[MOUSENAME_IDX])            
            latencyToFall = float(lines[LATENCY_TO_FALL_IDX])

            # IMPORTANT! WE ARE TRUNCATING LATENCIES TO 300, SINCE THAT'S DONE IN REFERENCE
            latencyToFall = 300 if latencyToFall > 300 else latencyToFall

            if mouseName in diff_dict:
                diff_dict[mouseName].append(latencyToFall)
            else:
                diff_dict[mouseName] = [latencyToFall]
    # convert it all to numpy array for faster comparison
    for key, val in diff_dict.items():
        diff_dict[key] = np.array(val)
    return diff_dict



# EXECUTION

if __name__ == "__main__":

    ref_dict = get_ref_dict()
    diff_dict = get_diff_dict()
    print(ref_dict)

    matches = find_matching_latencyToFall_entry(ref_dict=diff_dict, diff_dict=ref_dict, 
                                                error_margin=ERROR_MARGIN, include_same_name=SAVE_SAME_ENTRYNAME_PAIRS)
    identical_name_files = match_names_and_their_latencyToFall_entries(ref_dict=diff_dict, diff_dict=ref_dict)

    # merged_dict = matches
    # for k, val in identical_name_files.items(): merged_dict[k] = val

    # writing to csv file - first with the one that compares entries with different names
    with open(SAVE_DIFFERENT_NAME_MATCH_PATH, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)

        match_example = list(matches.values())[0]
        latencyToFallInRef_example = match_example[0]

        fields = ["Ref-Diff", "mouseName", ] + \
                ["LatencyToFall"] + [""] * (len(latencyToFallInRef_example) - 1) + \
                ["ErrorMargin"]

        # writing the fields  
        csvwriter.writerow(fields)
            
        # writing the data
        for mouseNameInRef, matched in matches.items():
            ref_row = [mouseNameInRef] + [round(val, SIG_FIGS) for val in matched[1]] + [ERROR_MARGIN]
            csvwriter.writerow(ref_row)

            diff_row = [matched[0]] + [round(val, SIG_FIGS) for val in matched[2]] + [""]
            csvwriter.writerow(diff_row)

            difference_of_two_files_row = ["Difference:"] + [round(val, SIG_FIGS) for val in matched[3]] + [""]
            csvwriter.writerow(difference_of_two_files_row)

    # then save that which compares entries with the same name
    with open(SAVE_SAME_NAME_PATH, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)

        match_example = list(identical_name_files.values())[0]
        latencyToFallInRef_example = match_example[0]

        fields = ["Ref-Diff", "mouseName", ] + \
                ["LatencyToFall"] + [""] * (len(latencyToFallInRef_example))

        # writing the fields  
        csvwriter.writerow(fields)
            
        # writing the data
        for mouseNameInRef, matched in identical_name_files.items():
            ref_row = [mouseNameInRef] + [round(val, SIG_FIGS) for val in matched[1]]
            csvwriter.writerow(ref_row)

            diff_row = [matched[0]] + [round(val, SIG_FIGS) for val in matched[2]]
            csvwriter.writerow(diff_row)

            difference_of_two_files_row = ["Difference:"] + [round(val, SIG_FIGS) for val in matched[3]]
            csvwriter.writerow(difference_of_two_files_row)