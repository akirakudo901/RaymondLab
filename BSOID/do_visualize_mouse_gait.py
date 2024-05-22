# Author: Akira Kudo
# Created: 2024/05/17
# Last Updated: 2024/05/22

import numpy as np

from feature_analysis_and_visualization.visualization.visualize_mouse_gait import visualize_locomotion_stats
from label_behavior_bits.preprocessing import filter_bouts_smaller_than_N_frames

LABEL_NPY = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\BSOID\results\feats_labels\Apr-08-2024_20230113135134_372301_m3_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_2060000_labels.npy"
label = np.load(LABEL_NPY)

print("Pre-filtering!")
visualize_locomotion_stats(df=None,
                            label=label,
                            bodyparts=None,
                            figure_name=None,
                            save_path=None,
                            interval=40*60*5,
                            locomotion_label=[29,30],
                            save_figure=False,
                            show_figure=True)

filt_label = filter_bouts_smaller_than_N_frames(label, n=5)
print("Post-filtering!")
visualize_locomotion_stats(df=None,
                            label=filt_label,
                            bodyparts=None,
                            figure_name=None,
                            save_path=None,
                            interval=40*60*5,
                            locomotion_label=[29,30],
                            save_figure=False,
                            show_figure=True)