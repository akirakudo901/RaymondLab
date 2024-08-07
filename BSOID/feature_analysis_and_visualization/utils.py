
import numpy as np
import re

BPT_ABBREV = {
    'snout' : 'sn',
    'rightforepaw' : 'rfp', 
    'leftforepaw' : 'lfp',
    'righthindpaw' : 'rhp',
    'lefthindpaw' : 'lhp',
    'tailbase' : 'tb',
    'belly' : 'bl'
}

# BELOW CODE WAS TAKEN FROM alimanfoo at:
# https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
def find_runs(x):
    """
    Find runs of consecutive items in an array.
    Returns: run_values, run_starts, run_lengths.
    """

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths
    
def get_mousename(filename : str):
    # Extracting information from the filename using regular expressions
    searched = re.search(r'(\d+)(_?)[mf](\d+)', filename)
    if searched:
       return searched[0].replace('_', '')
    else:
       return "No_Match"

def process_upper_and_lower_limit(limits : tuple):
    """
    Returns a sorted tuple of higher & lower values. 
    If limits contain a None entry, it is replaced by 
    float("-inf") for limits[0] and float("inf") for limits[1].
    Tuple passed with size != 2 raises an error.    
    """
    limits = list(limits)
    if len(limits) != 2: raise Exception("limits must be of size 2...")
    if limits[0] is None: limits[0] = float("-inf")
    if limits[1] is None: limits[1] = float("inf")
    return min(limits), max(limits)
