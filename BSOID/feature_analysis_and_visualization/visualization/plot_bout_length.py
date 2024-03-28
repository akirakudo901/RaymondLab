# Author: Akira Kudo
# Created: -
# Last updated: 2024/03/27

import os

import matplotlib.pyplot as plt
import numpy as np

from ..utils import find_runs

def plot_bout_length(labels : np.ndarray,
                     csv_name : str,
                     figure_save_dir : str,
                     show_figure : bool=True,
                     save_figure : bool=True,
                     use_logscale : bool=True):
  FIGSIZE_X, FIGSIZE_Y = 6, max(6, len(np.unique(labels))//2)
  YTICK_FONTSIZE = 7.5

  run_values, _, run_lengths = find_runs(labels)
  unique_labels = np.unique(run_values)

  R = np.linspace(0, 1, len(unique_labels))
  color = plt.cm.get_cmap("Spectral")(R)

  fig = plt.figure(facecolor='w', edgecolor='k', figsize=(FIGSIZE_X, FIGSIZE_Y))
  upperbound, lowerbound = max(run_lengths), min(run_lengths)

  for i, l in enumerate(unique_labels):
    plt.subplot(len(unique_labels), 1, i + 1)

    l_lengths = run_lengths[np.where(l == run_values)]

    num_bins = min(50, (upperbound - lowerbound)+1)
    plt.hist(l_lengths,
            bins=np.linspace(lowerbound, upperbound, num=num_bins),
            range=(lowerbound, upperbound),
            color=color[i])

    fig.suptitle("Length of features")
    plt.xlim(lowerbound, upperbound)
    plt.yticks(fontsize=YTICK_FONTSIZE)
    # if specified, use log scale for ticks
    if use_logscale and l_lengths.size != 0:
      plt.yscale('log')
    # add a label of which label group to each histogram
    plt.gca().set_title(f'{l}', loc='right', y=-0.2)

    if i < len(unique_labels) - 1:
      plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

  if save_figure:
      full_figurename = csv_name.replace(".csv", "")
      fig.savefig(os.path.join(figure_save_dir,
                              str.join('', (full_figurename,
                                            "_Logscale" if use_logscale else "",
                                            ".png" ))))
  if show_figure: plt.show()