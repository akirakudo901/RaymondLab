# Author: Akira Kudo
# Created: 2024/04/10
# Last Updated: 2024/04/12

import re

# some of the functions also exist in the BSOID part of this repo
# so I will clean this up someday

# also in BSOID/feature_analysis_and_visualization/utils.py
def get_mousename(filename : str):
    # Extracting information from the filename using regular expressions
    searched = re.search(r'(\d+)(_?)[mf](\d+)', filename)
    if searched:
       return searched[0].replace('_', '')
    else:
       return "No_Match"
    
bodypart_abbreviation_dict = {
    'snout' : 'snt', 'rightforepaw' : 'rfp', 'leftforepaw' : 'lfp',
    'righthindpaw' : 'rhp', 'lefthindpaw' : 'lhp', 'tailbase' : 'tlbs',
    'belly' : 'bll'
    }