# Author: Akira Kudo
# Created: 2024/04/03
# Last Updated: 2024/04/03

import os

import matplotlib.pyplot as plt
import numpy as np

from bsoid_io.utils import read_BSOID_labeled_csv

def plot_distribution_of_single_label_in_group_of_csv(groups : list, 
                                                      group_names : list,
                                                      labels_of_interest : list):
    for group, group_name in zip(groups, group_names):
        
        for loi in labels_of_interest:
            counts = []
            for csv in group:
                label = read_BSOID_labeled_csv(csv)[0]
                count = np.sum(label == loi)
                counts.append(count)
            plt.hist(counts, label=f'{loi}')
            plt.title(f"Label {loi} for Group {group_name}")
            plt.show()


YAC_FOLDER = r"C:\Users\mashi\Desktop\temp\YAC"
WT_FOLDER = r"C:\Users\mashi\Desktop\temp\WT"

yac_csvs = [os.path.join(YAC_FOLDER, file) for file in os.listdir(YAC_FOLDER) 
            if '.csv' in file]
wt_csvs  = [os.path.join(WT_FOLDER,  file) for file in os.listdir(WT_FOLDER )
            if '.csv' in file]

plot_distribution_of_single_label_in_group_of_csv(
    groups=[yac_csvs + wt_csvs], 
    group_names=["YAC_AND_WT"], 
    labels_of_interest=[32, 38]
)