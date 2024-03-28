# Python file to display training results, from a log.txt file.

import os
import re

import matplotlib.pyplot as plt

ITERATION_IDX_IN_SPLIT = 3
LOSS_IDX_IN_SPLIT = 5
LR_IDX_IN_SPLIT = 7

def parse_lines(list_of_time_stamps):
    runs = {}
    run_count = 1

    iterations, losses, lrs = [], [], []
    for line in list_of_time_stamps:
        if (not isinstance(line, str)) or ("iteration" not in line) or \
           ("loss" not in line) or ("lr" not in line):
            continue
        else:
            split_by_spaces = re.split(r"\s", line)
            # have multiple entries at times - we seperate those
            iter_val = int(split_by_spaces[ITERATION_IDX_IN_SPLIT])
            loss_val = float(split_by_spaces[LOSS_IDX_IN_SPLIT])
            lr_val = float(split_by_spaces[LR_IDX_IN_SPLIT])

            if len(iterations) > 0 and iter_val < iterations[-1]: # if iter_val goes down, it's new
                runs[run_count] = {"iterations" : iterations, 
                                   "losses" : losses, 
                                   "lrs" : lrs}
                iterations, losses, lrs = [], [], []
                run_count += 1

            iterations.append(iter_val)
            losses.append(loss_val)
            lrs.append(lr_val)
    runs[run_count] = {"iterations" : iterations, 
                       "losses" : losses, 
                       "lrs" : lrs}
    
    return runs

# os.chdir(r"c:/Users/mashi/Desktop/train_loss_visualization")
os.chdir(r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/dlc-models/iteration-0/Q175-D2Cre Open Field Males BrownJan12-trainset95shuffle1/train")

f = open("log.txt", "r")
lines = f.readlines()

runs = parse_lines(lines)

for i in range(1, len(runs.keys()) + 1):
    run = runs[i]; iterations = run["iterations"]; losses = run["losses"]

    print(f"Run {i} has {len(iterations)} entries.")

    plt.figure(figsize=(10,6))
    plt.plot(iterations, losses)
    plt.yscale('log')
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.title(f"Iteration {i}")
    plt.grid(True)
    plt.show()
