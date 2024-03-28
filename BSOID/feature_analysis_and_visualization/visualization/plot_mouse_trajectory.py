# Author: Akira Kudo
# Created: -
# Last updated: 2024/03/28

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_extraction.BSOID_code.adp_filt import adp_filt
from feature_extraction.extract_label_and_feature_from_csv import FEATURE_SAVING_FOLDERS

def plot_mouse_trajectory(csvpath : str, figureName : str,
                          start : int=0, end : int=None,
                          bodypart : str="tailbase",
                          show_figure : bool=True,
                          save_figure : bool=True,
                          save_path : str=FEATURE_SAVING_FOLDERS):
  """
  Plots how the mouse moved in the cage over time, with points progressively
  increasing in shade darkness for later time points.
  """
  raw_data = pd.read_csv(csvpath, low_memory=False)
  filtered_data, _ = adp_filt(currdf=raw_data, pose=list(range(raw_data.shape[1]-1)))
  # bodyparts in order: snout, rightforepaw, leftforepaw, righthindpaw,
  #                     lefthindpaw, tailbase, belly
  bodyparts = ["snout", "rightforepaw", "leftforepaw", "righthindpaw",
               "lefthindpaw", "tailbase", "belly"]
  if bodypart.lower() not in bodyparts:
    bodypart = "tailbase"
    print("Given bodypart is unexpected - using tailebase instead!")
  # get the bodypart index within the bodypart header
  bp_idx = bodyparts.index(bodypart)
  # get the x and y position of the bodypart of interest
  bp_x_idx, bp_y_idx = bp_idx*2, bp_idx*2 + 1
  # get the x/y data out of the filtered data
  X,Y = filtered_data[:,bp_x_idx], filtered_data[:,bp_y_idx]

  if end is None: end = len(X)
  plot_figure_over_time(X, Y, start, end, figureName, show_figure, save_figure, save_path)

# Plots joy stick position over time, with points progressively increasing in
# shade darkness for later time points
def plot_figure_over_time(X : np.ndarray, Y : np.ndarray,
                          start : int, end : int, figureName : str,
                          show_figure : bool=True,
                          save_figure : bool=True,
                          save_path : str=FEATURE_SAVING_FOLDERS):
  # clip end
  end = min(end, len(X))
  # render chunks of the image at once for speed up
  num_rendering_chunk = 1000
  chunk_size = (end - start) // 1000
  blueColor = plt.cm.Blues(np.linspace(0.1,1,num_rendering_chunk))

  fig,ax = plt.subplots(figsize=(6,6))
  for k in range(num_rendering_chunk):
    render_start = start + chunk_size * k
    render_end   = start + chunk_size * (k + 1) - 1
    render_end = min(render_end, end)
    ax.plot(X[render_start:render_end], Y[render_start:render_end],
            color=blueColor[k])
  plt.xlabel('X'); plt.ylabel('Y')
  plt.title(figureName)
  plt.xlim(min(X), max(X)); plt.ylim(min(Y), max(Y))

  if save_figure:
    if not os.path.exists(save_path):
      print(f"{os.path.basename(save_path)} did not exist - created it!")
      os.mkdir(save_path)
    print(f"Saving {figureName} to {save_path}!")
    plt.savefig(os.path.join(save_path, figureName + '_figure.png'))

  if show_figure: plt.show()