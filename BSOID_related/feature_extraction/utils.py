# Author: Akira Kudo
# Created: -
# Last Updated: 2024/03/20

from enum import Enum
import itertools
from typing import List

# define an enumeration for body parts
class Bodypart(Enum):
  SNOUT = 0
  RIGHTFOREPAW = 1
  LEFTFOREPAW = 2
  RIGHTHINDPAW = 3
  LEFTHINDPAW = 4
  TAILBASE = 5
  BELLY = 6
  
# some useful functions to internally align names
def relative_placement_name(bp1 : Bodypart, bp2 : Bodypart, short=False):
  # swap values to ensure a strict ordering
  if bp1.value > bp2.value: bp1, bp2 = bp2, bp1
  if short:
     return "{}_{}_dist".format(bp1.name, bp2.name)
  else:
     return "Relative {} to {} placement".format(bp1.name, bp2.name)

def relative_angle_name(bp1 : Bodypart, bp2 : Bodypart, short=False):
  # swap values to ensure a strict ordering
  if bp1.value > bp2.value: bp1, bp2 = bp2, bp1
  if short:
     return "{}_{}_angle".format(bp1.name, bp2.name)
  else:
     return "Relative angle from {} to {}".format(bp1.name, bp2.name)

def displacement_name(bp : Bodypart, short=False):
   if short:
      return "{}_disp".format(bp.name)
   else:
      return "Displacement of {}".format(bp.name)

# Given features are extracted in an internal order as read from the csv that
# holds the original data, this generates a good guess on which feature index
# in the feature vector created by the above function of
# extract_pregenerated_labels_and_compute_features
# correspond to which feature in English.
def generate_guessed_map_of_feature_to_data_index(bodyparts : List[Bodypart], short=False):
  """
  :param List[Bodypart] bodyparts: A list of Bodypart Enum objects indicating
  which body part are considered for feature generation.
  :param bool short: Whether to shorten the returned names.
  :return Dict[str, int]: A dictionary mapping a set of standardized names
  generated from body parts with their guessed index. The standardized names are
  generated using naming functions defined right above.
  """
  bodypart_pairs = list(itertools.combinations(bodyparts, 2))

  educated_guess = dict()
  i = 0
  for bp1, bp2 in bodypart_pairs:
      educated_guess[relative_placement_name(bp1, bp2, short)] = i
      i += 1
  for bp1, bp2 in bodypart_pairs:
      educated_guess[relative_angle_name(bp1, bp2, short)] = i
      i += 1
  for bp in bodyparts:
      educated_guess[displacement_name(bp, short)] = i
      i += 1
  return educated_guess