# Author: Akira Kudo
# Created: 2024/03/11
# Last updated: 2024/03/11

import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

ITERATION_COL_TITLE = 'Training iterations:'
TRAINERROR_COL_TITLE = ' Train error(px)'
TESTERROR_COL_TITLE = ' Test error(px)'

def visualize_train_test_loss_from_evaluation_csv(csv_path : str, use_logscale : bool=False):
    """
    Visualize the training and test losses stored in a csv after 
    evaluation of multiple network snapshots. These would often 
    be called 'CombinedEvaluation-results.csv'.
    :param str csv_path: Path to csv that holds the info.
    """
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots (figsize=(6,6))
    ax.plot(df[ITERATION_COL_TITLE], df[TRAINERROR_COL_TITLE], label="Train error")
    ax.plot(df[ITERATION_COL_TITLE],  df[TESTERROR_COL_TITLE], label= "Test error")
    if use_logscale:
        plt.yscale('log')
    ax.legend()
    plt.show()
    
if __name__ == "__main__":
    yscale_type = input("Would you want to use a log-scale? y/n").lower() in ['y', 'yes']
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(initialfile="/media/Data/Raymond Lab") # show an "Open" dialog box and return the path to the selected file
    CSV_PATH = filename
    
    # CSV_PATH = "/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/evaluation-results/iteration-1/CombinedEvaluation-results.csv"
    visualize_train_test_loss_from_evaluation_csv(CSV_PATH, use_logscale=yscale_type)
