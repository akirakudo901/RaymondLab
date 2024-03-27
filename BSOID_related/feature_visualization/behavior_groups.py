# Author: Akira Kudo
# Created: 2024/03/27
# Last updated: 2024/03/27

"""
Given bigger 'groups' of behavior above labels, such as:
- rest / locomotion / right head turn / left head turn / ...
Visually and statistically compare different features,
1) between groups, and
2) within the group themselves.
"""

import yaml

YAML_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID_related\feature_visualization\behavior_groups.yml"

class BehaviorGrouping:

    def __init__(self, network_name : str, yaml_path : str=YAML_PATH):
        self.label_to_groups = {}
        self.load_behavior_groupings(network_name, yaml_path)
        
    def load_behavior_groupings(self, network_name : str, yaml_path : str=YAML_PATH):
        """
        Loads the behavior groupings from a yaml file - which is groups 
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
        self.groupings = yaml_dict['networks'][network_name]
        # given self.groupings maps behavior groups to labels, 
        # also get the opposite 
        if self.label_to_groups: self.label_to_groups = {} # reset if exists
        for group, labels in self.groupings.items():
            for l in labels:
                self.label_to_groups[l] = group
        return self.groupings

    def label_to_behavioral_group(self, label : int):
        """
        :returns str: Returns the corresponding behavioral group string, or 
        "Not found" if not in the dictionary.
        """
        if label not in self.label_to_groups.keys():
            print(f"Provided label {label} isn't in the behavioral groupings!")
            return "Not found"
        else:
            return self.label_to_groups[label]

    def behavioral_group_to_label(self, behavioral_group : str):
        """
        :returns List[int]: Returns the corresponding labels, or 
        an empty list if not in the dictionary.
        """
        if behavioral_group not in self.groupings.keys():
            print(f"Provided behavioral_group {behavioral_group} isn't in the behavioral groupings!")
            return []
        else:
            return self.groupings[behavioral_group]


if __name__ == "__main__":
    NETWORK_NAME = 'Feb-23-2023'
    bg = BehaviorGrouping(network_name=NETWORK_NAME, yaml_path=YAML_PATH)
    groupings = bg.load_behavior_groupings(network_name=NETWORK_NAME, yaml_path=YAML_PATH)
    print(groupings)
    print(f"32: {bg.label_to_behavioral_group(32)}")
    print(f"38: {bg.label_to_behavioral_group(38)}")