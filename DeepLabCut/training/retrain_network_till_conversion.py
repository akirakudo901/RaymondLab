# Author: Akira Kudo
# Created: 2024/05/09
# Last Updated: 2024/05/09

import deeplabcut

if __name__ == "__main__":
    SHUFFLE = 1
    MAX_SNAPSHOTS = 99
    DISPLAY_ITERS = 1000
    SAVEITERS = 50000
    MAXITERS = 3030000

    # DON'T FORGET TO SET SNAPSHOT INDEX TO ALL TO EVALUATE ALL CREATED SNAPSHOTS!
    PLOT_SCORE_MAPS = False
    SHUFFLES = [1]
    COMPARISON_BODYPARTS = "all"

    PATH_CONFIG_FILE = "/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/config.yaml"

    deeplabcut.train_network(PATH_CONFIG_FILE, 
                            shuffle=SHUFFLE, 
                            displayiters=DISPLAY_ITERS,
                            saveiters=SAVEITERS,
                            max_snapshots_to_keep=MAX_SNAPSHOTS,
                            maxiters=MAXITERS)

    deeplabcut.evaluate_network(PATH_CONFIG_FILE,
                                Shuffles=SHUFFLES,
                                plotting=PLOT_SCORE_MAPS,
                                show_errors=True,
                                comparisonbodyparts=COMPARISON_BODYPARTS
                                )