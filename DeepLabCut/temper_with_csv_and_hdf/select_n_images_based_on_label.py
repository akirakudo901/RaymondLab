# Author: Akira Kudo

# Obtain a dataset that merges the original csv with the storage labels,
# then select a set number of the 0 and 1 label so that the total becomes 50.
# Finally, save a reduced version.

from typing import List
import os

import pandas as pd

def select_n_images_based_on_label(coord_csv_path : str, 
                                   label_csv_path : str, 
                                   label : List[int], 
                                   n : List[int], 
                                   sample_seed : int=None):
    coord_df = pd.read_csv(coord_csv_path)
    label_df = pd.read_csv(label_csv_path, index_col=[0])
    coord_header = coord_df.loc[:1]
    # merge dataframes
    merged_df = coord_df.merge(label_df, how="outer", right_on=["origindir", "imagename"], 
                   left_on=["Unnamed: 1", "Unnamed: 2"])
    # select n rows of the dataframe with corresponding label
    selected_df = merged_df[merged_df['category'] == label]
    num_sample = min(n, selected_df.shape[0])
    if sample_seed is None:
        selected_df = selected_df.loc[:, coord_df.columns].sample(n=num_sample)
    else:
        selected_df = selected_df.loc[:, coord_df.columns].sample(n=num_sample, random_state=sample_seed)
    selected_df = pd.concat([coord_header, selected_df])

    return selected_df

if __name__ == "__main__":
    DATA_DIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\previous\B-SOID STUFF"
    COORD_CSV_PATH = os.path.join(DATA_DIR, r"20220211070325_301533_f3_\CollectedData_Judy.csv")
    LABEL_CSV_PATH = os.path.join(DATA_DIR, r"20220211070325_301533_f3__storage.csv")
    LABEL = 0 #0 : Wall rear, 1 : Grooming, 2 : Other
    N = 5

    selected_df = select_n_images_based_on_label(COORD_CSV_PATH, LABEL_CSV_PATH, LABEL, N)
    print(selected_df)
