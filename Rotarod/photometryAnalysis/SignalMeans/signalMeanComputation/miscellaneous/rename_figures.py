# USED CHATGPT 3.5 FOR THE BASIS OF THIS CODE!

import os
import re

def convert_filename(filename):
    if not filename.endswith(".jpg"):
        return None
    elif filename.startswith("370261_m1_NOT_372301_4_2"):
        remaining = filename[len("370261_m1_NOT_372301_4_2") : ]
        newname = "370261_m1_NOT_372301_11" + remaining
        print("Special case: " + filename + " -> " +  newname)
        return newname
    elif filename.startswith("m3_1_1"):
        remaining = filename[len("m3_1_1") : ]
        newname = "m3_1" + remaining
        print("Special case: " + filename + " -> " +  newname)
        return newname

    # Define a regular expression pattern to match the naming convention
    pattern = r'(\d+)_?[mf](\d+)_(\d+)_(\d+) (.*)\.jpg'
    outlierpattern = r'(\d+)_?[mf](\d+)_(\d+)_(\d+)[_-](\d+) (.*)\.jpg'
    
    newpattern = r'(\d+)_?[mf](\d+)_(\d+)(-\d+)? (.*)\.jpg'

    
    # Use regular expression to extract components of the filename)
    match = re.match(pattern, filename)
    
    if match:
        # Extract components
        somename = match.group(1)
        N = match.group(2)
        P = int(match.group(3))
        Q = int(match.group(4))
        other = match.group(5)

        # Compute R
        R = (P - 1) * 3 + Q

        # Construct the new filename
        new_filename = f"{somename}_m{N}_{R} {other}.jpg"
        return new_filename
    elif re.match(newpattern, filename):
        return None
    elif match := re.match(outlierpattern, filename):
        # Extract components
        somename = match.group(1)
        N = match.group(2)
        P = int(match.group(3))
        Q = int(match.group(4))
        WITHIN_TRIAL = int(match.group(5))
        other = match.group(6)

        # Compute R
        R = (P - 1) * 3 + Q

        # Construct the new filename
        new_filename = f"{somename}_m{N}_{R}-{WITHIN_TRIAL} {other}.jpg"
        print("Outlier: " + filename + "; new name: " + new_filename)
        return new_filename
    else:
        print("Filename does not match the expected format: " + filename)
        return None

def convert_everything_in_folder(d):
    for f in os.listdir(d):
        new_filename = convert_filename(f)
        if new_filename:
            full_old_name = os.path.join(d, f)
            full_new_name = os.path.join(d, new_filename)
            # os.rename(full_old_name, full_new_name)

DIR = r"C:\\Users\\mashi\Desktop\\RaymondLab\\Experiments\\Rotarod\\photometryAnalysis\SignalMeans\Akira_photometry_rotarod_scripts\\Ellen script\\results"

for dir_object in os.listdir(DIR):
    fullpath_to_dir = os.path.join(DIR, dir_object)
    if os.path.isdir(fullpath_to_dir):
        convert_everything_in_folder(fullpath_to_dir)