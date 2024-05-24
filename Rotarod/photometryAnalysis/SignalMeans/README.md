# Computing Signal Means

This folder holds code used to systematically analyze the photometry signals for rotarod. 
It builds on top of Ellen's script, **photometry_rotarodaccelerating_redandgreen_updatedanalysis**.

Compute mean of GCAMP/RCAMP signals
-

If you want to **compute the normal/shifted means of GCAMP/RCAMP signals** from folder that hold Rotarod Photometry data  (e.g. **Raymond Lab\2 Colour ...\B6-Q175 Mice -- 6 and 10 months\D1-GCAMP and D2-RCAMP (regular)\Rotarod**)

Then, take a look at:
* ```signalMeanComputation > apply_photometry_analysis_to_many.m```

If you need to **debug anything**, using the **single-file analysis version** is handy:
* ```signalMeanComputation > analyze_one_file.m```

Once you have analyzed multiple file, it will occur to you that:
* Some folder have typos, making the csv weird to read.
  * You could then use ```merge_two_mice_data_entries``` to merge the csv parts that are likely typo, from:
    * ```signalMeanComputation > miscellaneous > merge_means_data_based_on_trial_holes.py```