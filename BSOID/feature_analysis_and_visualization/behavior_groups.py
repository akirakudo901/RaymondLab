# Author: Akira Kudo
# Created: 2024/03/27
# Last updated: 2024/08/30

"""
Given bigger 'groups' of behavior above labels, such as:
- rest / locomotion / right head turn / left head turn / ...
Visually and statistically compare different features,
1) between groups, and
2) within the group themselves.
"""

import os
from typing import Union

import yaml

YAML_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\SortedForMarja\MetaLabel\behavior_groups_Akira.yml"

class BehaviorGrouping:

    def __init__(self, network_name : str, yaml_path : str=YAML_PATH):
        self.label_to_groups_str, self.label_to_groups_int = {}, {}
        self.load_behavior_groupings(network_name, yaml_path)
        
    def load_behavior_groupings(self, network_name : str, yaml_path : str=YAML_PATH):
        """
        Loads the behavior groupings from a yaml file - which groups 
        behavior labels from a B-SOID network into subgroups based on 
        visual inspection.
        Groups would be of higher abstraction, such as 'rest', 'locomotion', etc.

        :param str network_name: The name of the network to load the groupings,
        e.g. 'Feb-23-2023'
        :param str yaml_path: The path to yaml storing the groupings, 
        defaults to YAML_PATH.
        :returns Dict: A dictionary which maps names of groupings to a list 
        of labels that belong to them.
        """
        with open(yaml_path, 'r') as f:
            document = "\n".join(f.readlines())
        
        self.network_name = network_name
        self.yaml_path = yaml_path

        yaml_dict = yaml.safe_load(document)
        self.groupings_str = yaml_dict['networks'][network_name]
        # also define integer label of groupings & back-and-forth conversion
        # *use the fact that keys, values and items are traversed in the same order
        self.grouping_str_to_grouping_int = dict([
            (grouping, i) for i, grouping in enumerate(self.groupings_str.keys())
            ])
        self.grouping_int_to_grouping_str = dict([
            (i, grouping) for i, grouping in enumerate(self.groupings_str.keys())
            ])
        self.groupings_int = dict([
            (i, lst) for i, lst in enumerate(self.groupings_str.values())
            ])

        # given self.groupings maps behavior groups to labels, 
        # also get the opposite 
        if self.label_to_groups_str: self.label_to_groups_str = {} # reset if exists
        if self.label_to_groups_int: self.label_to_groups_int = {} # reset if exists
        for i, group_with_label in enumerate(self.groupings_str.items()):
            group, labels = group_with_label
            for l in labels:
                self.label_to_groups_str[l] = group
                self.label_to_groups_int[l] = i
        return self.groupings_str

    def label_to_behavioral_group_str(self, label : int):
        """
        :returns str: Returns the corresponding behavioral group string, or 
        "Not found" if not in the dictionary.
        """
        if label not in self.label_to_groups_str.keys():
            print(f"Provided label {label} isn't in the behavioral groupings!")
            return "Not found"
        else:
            return self.label_to_groups_str[label]
    
    def label_to_behavioral_group_int(self, label : int):
        """
        :returns str: Returns the corresponding behavioral group integer, or 
        -1 if not in the dictionary.
        """
        if label not in self.label_to_groups_int.keys():
            print(f"Provided label {label} isn't in the behavioral groupings!")
            return -1
        else:
            return self.label_to_groups_int[label]

    def behavioral_group_to_label(self, behavioral_group : Union[str, int]):
        """
        :returns List[int]: Returns the corresponding labels, or 
        an empty list if not in the dictionary.
        """
        if isinstance(behavioral_group, int):
            if behavioral_group in self.groupings_int.keys():
                return self.groupings_int[behavioral_group]
        elif isinstance(behavioral_group, str):
            if behavioral_group in self.groupings_str.keys():
                return self.groupings_str[behavioral_group]
        # otherwise, it wasn't found
        print(f"Provided behavioral_group {behavioral_group} isn't in the behavioral groupings!")
        return []
    
    def get_behavior_groups(self):
        """
        :returns List[str]: Returns the list of string corresponding
        to all behavior groups in this.
        """
        return list(self.groupings_str.keys())
        
    


if __name__ == "__main__":
    NETWORK_NAME = 'Feb-23-2023'
    bg = BehaviorGrouping(network_name=NETWORK_NAME, yaml_path=YAML_PATH)
    groupings = bg.load_behavior_groupings(network_name=NETWORK_NAME, yaml_path=YAML_PATH)
    print(groupings)
    print(f"32: {bg.label_to_behavioral_group_str(32)}; {bg.label_to_behavioral_group_int(32)}")
    print(f"38: {bg.label_to_behavioral_group_str(38)}; {bg.label_to_behavioral_group_int(38)}")