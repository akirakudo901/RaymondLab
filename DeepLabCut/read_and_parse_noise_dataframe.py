# Author: Akira Kudo
# Created: 2024/04/11
# Last Updated: 2024/04/24

import os
import re

import numpy as np
import pandas as pd

from visualization.visualize_speed_of_bodypart import visualize_angle_of_bodypart_from_csv, visualize_likelihood_of_bodypart_from_csv, visualize_property_of_bodypart_from_csv, visualize_speed_of_bodypart_from_csv
from temper_with_csv_and_hdf.data_filtering.identify_paw_noise import identify_bodypart_noise_by_impossible_speed

NOISE_NUMBER_TO_BODYPART_MAP = {
    0 : 'snout',
    1 : 'rightforepaw',
    2 : 'leftforepaw',
    3 : 'righthindpaw',
    4 : 'lefthindpaw',
    5 : 'tailbase',
    6 : 'belly'
}

def read_and_parse_noise_dataframe(df_csv : str, start : int=None, end : int=None):
    # get the csv indicating which frames are noise, sorting it as needed
    noise_df = pd.read_csv(df_csv, index_col=0); noise_df = noise_df[:-1]
    image_ids = noise_df.imagename.str.extract(r'(\d+)', expand=False)
    noise_df.insert(noise_df.shape[1], column="image_id", value=image_ids.astype(np.int64))
    noise_df.sort_values(by="image_id", inplace=True)
    noise_df.reset_index(inplace=True, drop=True)

    # obtain the 'category' indicating what kind of noise this frame is
    bodyparts2noise = {}
    categories = noise_df['category'].to_numpy()

    for bpt_lbl in np.unique(categories):
        contained_labels = [entry for entry in re.findall('[-]?\d+', bpt_lbl)]
        # we ignore 'empty' entries, that are '-1'
        if contained_labels[0] == "-1":
            pass
        # if we found a lonely entry, get the bool array corresponding
        elif len(contained_labels) == 1:
            bpt_str = NOISE_NUMBER_TO_BODYPART_MAP[int(bpt_lbl)]
            bodyparts2noise[bpt_str] = (categories == bpt_lbl)
        # otherwise, iterate and update the dictionary accoprdingly
        else:
            for bpt_lbl_lone in contained_labels:
                bpt_str = NOISE_NUMBER_TO_BODYPART_MAP[int(bpt_lbl_lone)]
                bodyparts2noise[bpt_str] = np.logical_or(
                    (categories == bpt_lbl), # first is where bpt_lbl itself is located
                    # second is where bpt_lbl_lone itself is contained
                    bodyparts2noise.get(bpt_str, np.full((len(categories),), False))
                    )

    # truncating result as wished
    if start is None: start = 0
    if end is None: end = len(categories)

    for k in bodyparts2noise.keys():
        val = bodyparts2noise[k]
        bodyparts2noise[k] = val[start:end+1]

    return bodyparts2noise

def sanity_check_read_and_parse_noise_dataframe():
    SNCHECK_CSV = r"C:\Users\mashi\Desktop\RaymondLab\Experiments\Openfield\3part2 BsoidAnalysis\tkteach\ds\sanity_check_storage.csv"
    noise_map = read_and_parse_noise_dataframe(SNCHECK_CSV)
    expected_map = {
        'snout' :        np.array([False,False,False,False,True, False,False,False,False,True ,True]),
        'rightforepaw' : np.array([False,True, False,False,True, False,False,False,False,False,True]),
        'leftforepaw'  : np.array([False,False,True ,False,False,False,False,False,False,False,True]),
        'righthindpaw' : np.array([False,False,False,True ,False,False,False,False,True ,False,True]),
        'lefthindpaw'  : np.array([False,False,False,False,False,False,True ,True ,False,False,True]),
        'tailbase'     : np.array([False,False,False,True ,False,False,False,False,False,False,False]),
    }
    # compare number of entries
    assert len(noise_map.keys()) == len(expected_map.keys())
    # then the keys
    for k in noise_map.keys():
        assert k in expected_map.keys()
        # while comparing values
        assert np.array_equal(noise_map[k], expected_map[k])

if __name__ == "__main__":
    CSV_FOLDER = r"C:\Users\mashi\Desktop\temp\Q175\csvs"
    # r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\B-SOID\Q175 Open Field CSVs\WT\snapshot2060000"
    CSV_FILENAME = "20220228223808_320151_m1_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_2060000.csv"
    CSV_PATH = os.path.join(CSV_FOLDER, CSV_FILENAME)

    START, END = 100, 400

    NOISE_CSV = r"C:\Users\mashi\Desktop\RaymondLab\Experiments\Openfield\3part2 BsoidAnalysis\tkteach\ds\labeled_extracted_storage.csv"

    bodyparts2noise = read_and_parse_noise_dataframe(NOISE_CSV, start=START, end=END)

    # finally visualize the result
    # visualize_speed_of_bodypart_from_csv(
    #     csv_path=CSV_PATH, 
    #     bodyparts=[
    #         'snout',
    #         'rightforepaw', 'leftforepaw',
    #         # 'righthindpaw', 
    #         # 'lefthindpaw',
    #         # 'tailbase', 'belly'
    #         ],
    #     start=START, end=END,
    #     bodyparts2noise=bodyparts2noise
    # )

    # visualize_angle_of_bodypart_from_csv(
    #     csv_path=CSV_PATH, 
    #     bodyparts=[
    #         'snout',
    #         'rightforepaw', 'leftforepaw',
    #         # 'righthindpaw', 
    #         # 'lefthindpaw',
    #         # 'tailbase', 
    #         # 'belly'
    #         ],
    #     start=START, end=END,
    #     bodyparts2noise=bodyparts2noise
    # )

    BODYPARTS = [
            'snout',
            'rightforepaw', 'leftforepaw',
            'righthindpaw', 
            'lefthindpaw',
            'tailbase', 
            'belly'
            ]

    # visualize_property_of_bodypart_from_csv(
    #     csv_path=CSV_PATH, 
    #     bodyparts=BODYPARTS, 
    #     flag=25, 
    #     start=START, end=END,
    #     bodyparts2noise=bodyparts2noise
    # )

    visualize_likelihood_of_bodypart_from_csv(
        csv_path=CSV_PATH, 
        bodyparts=BODYPARTS, 
        start=START, end=END,
        bodyparts2noise=bodyparts2noise
        )

    # also calcualte bodyparts2noise from impossible speed & location
    impossible_move_df = identify_bodypart_noise_by_impossible_speed(
        dlc_csv_path=CSV_PATH, bodyparts=BODYPARTS, start=START, end=END
    )

    bodyparts2noise_from_impossible_speed = dict(
        [( bpt, impossible_move_df[(bpt, "loc_wrng")].to_numpy() ) 
         for bpt in BODYPARTS]
    )

    # visualize_property_of_bodypart_from_csv(
    #     csv_path=CSV_PATH, 
    #     bodyparts=BODYPARTS, 
    #     flag=25, 
    #     start=START, end=END,
    #     bodyparts2noise=bodyparts2noise_from_impossible_speed
    # )

    visualize_likelihood_of_bodypart_from_csv(
        csv_path=CSV_PATH, 
        bodyparts=BODYPARTS, 
        start=START, end=END,
        bodyparts2noise=bodyparts2noise_from_impossible_speed
        )