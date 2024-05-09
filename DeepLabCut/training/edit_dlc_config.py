# Author: Akira Kudo
# Created: 2024/05/09
# Last Updated: 2024/05/09

import os

import yaml

def edit_config(config_path : str,
                **kwargs):
    """
    Edit a DeepLabCut project config by passing keyword arguments
    corresponding to entries in a possible config document.
    Given keyword arguments must be already part of the given 
    config file, or will be ignored.
    This can be used for both config and pose_cfg.
    
    :param str config_path: Path to DLC config yaml.
    :param **kwargs: Any keyword argument to be updated inside the
    given DLC config. Keys must be already part of the config, or 
    will be ignored.
    """
    with open(config_path, 'r') as file:
        cnfg = yaml.safe_load(file)
    
    for key, val in kwargs.items():
        if key in cnfg.keys():
            cnfg[key] = val
        else:
            print(f"Key {key} isn't already in the config file, being ignored.")
    
    with open(config_path, 'w') as file:
        yamlized_cnfg = yaml.dump(cnfg)
        file.write(yamlized_cnfg)

if __name__ == "__main__":
    CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "example_config.yaml")
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Pre-edit: Task - {config['Task']}; TrainingFraction - {config['TrainingFraction']}")

    edit_config(CONFIG_PATH, Task='CHANGED!', TrainingFraction='100%', this_param_does_not_exist='32')

    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Post-edit: Task - {config['Task']}; TrainingFraction - {config['TrainingFraction']}")

    edit_config(CONFIG_PATH, Task='Example', TrainingFraction='80')