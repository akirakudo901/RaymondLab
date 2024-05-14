# Author: Akira Kudo
# Created: 2024/05/09
# Last Updated: 2024/05/12

import os

import deeplabcut
import numpy as np
import pandas as pd
import yaml

TRAINING_ITERATIONS = "Training iterations:"
SHUFFLE_NUMBER = "Shuffle number"

def train_with_early_stopping(
        config : str, # DLC original parameter
        earlystop_obs_num : int=None,
        use_filtered_error : bool=False,
        # DLC original parameters
        shuffle=1,
        trainingsetindex=0, 
        max_snapshots_to_keep=None, 
        displayiters=None, 
        saveiters=None, 
        maxiters=None, 
        allow_growth=True, 
        gputouse=None, 
        autotune=False, 
        keepdeconvweights=True, 
        modelprefix=''
        ):
    
    error_col = "Test error with p-cutoff" if use_filtered_error else "Test error(px)"

    # access pose_cfg.yaml
    pose_cfg_dict, _ = fetch_pose_cfg_given_config(config_path=config, shuffle=shuffle)    
    # if saveiters is given as None, load it (we need it for early stopping)
    if saveiters is None: saveiters = pose_cfg_dict['save_iters']
    # we also need the starting iteration number from pose_cfg
    init_weights_name = os.path.basename(pose_cfg_dict["init_weights"])
    # either the snapshot number we initialize weights from, or 0 if init_weights isn't a snapshot
    start_iter = int(init_weights_name.replace('snapshot-','')) if 'snapshot-' in init_weights_name else 0
    
    # we train a network specified as usual, except we train them for saveiters at a time
    end_iter = start_iter + saveiters
    while end_iter <= maxiters:
        deeplabcut.pose_estimation_tensorflow.training.train_network(
            maxiters=end_iter,
            # everything else is called according to what's passed
            config=config, shuffle=shuffle, trainingsetindex=trainingsetindex, 
            max_snapshots_to_keep=max_snapshots_to_keep, displayiters=displayiters, 
            saveiters=saveiters, allow_growth=allow_growth, gputouse=gputouse, 
            autotune=autotune, keepdeconvweights=keepdeconvweights, 
            modelprefix=modelprefix
            )

        # then we evaluate the newest snapshot saved (which just got trained)
        deeplabcut.pose_estimation_tensorflow.core.evaluate.evaluate_network(
            config,
            Shuffles=[shuffle], trainingsetindex=trainingsetindex,
            plotting=False, show_errors=True, comparisonbodyparts='all',
            gputouse=gputouse, rescale=False, modelprefix='')

        # we observe the latest snapshot evaluation losses, and if we find consecutive
        # losses in accuracy that happen earlystop_obs_num in a row, we stop
        csv_df, _ = fetch_evaluation_csv_given_config(config_path=config)
        df_curr_training = csv_df[csv_df[SHUFFLE_NUMBER] == shuffle]
        
        losses = df_curr_training[
            (df_curr_training[TRAINING_ITERATIONS] <= end_iter) & 
            (df_curr_training[TRAINING_ITERATIONS] >= end_iter - earlystop_obs_num * saveiters)
        ]
        losses = losses.sort_values(by=[TRAINING_ITERATIONS], ascending=True)

        change_in_error = np.diff(losses[error_col])
        # if all changes in error between consecutive saveiters are increasing, stop
        if np.all(change_in_error > 0):
            print(f"We've seen {earlystop_obs_num} consecutive increase in error - stopping early!")
            break

        end_iter += saveiters
    
    csv_df, _ = fetch_evaluation_csv_given_config(config_path=config)
    df_curr_training = csv_df[csv_df[SHUFFLE_NUMBER] == shuffle]
    df_curr_training = df_curr_training.sort_values(by=[TRAINING_ITERATIONS], ascending=True)
    
    min_error_value = np.min(df_curr_training[error_col])
    min_error_iteration = df_curr_training.iloc[np.argmin(df_curr_training[error_col]), :][TRAINING_ITERATIONS]
        
    print(f"Best test error (px) {'with cutoff ' if use_filtered_error else ''}was seen with " + 
          f"network at iteration: {min_error_iteration}, with error: {min_error_value}.")

        
def fetch_pose_cfg_given_config(config_path : str, shuffle : int):
    """
    Fetches the pose_cfg.yaml file path & content given a config.yaml and 
    shuffle number for a given DLC project. pose_cfg is located under:
    CONFIG_PARENT_DIR > dlc-models > iteration-N > NETWORK_NAME > train > pose_cfg.yaml

    :param str config_path: Path to config.yaml of the DLC project.
    :param int shuffle: Shuffle number for the specific training - not in config.

    :returns pose_cfg_dict, pose_cfg_path: pose_cfg read into a dict, as well as 
    its path.
    """
    # CONFIG_PARENT_DIR > dlc-models > iteration-N > NETWORK_NAME > train > pose_cfg.yaml
    config_parent_dir = os.path.dirname(config_path)
    
    with open(config_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    iteration, Task = cfg_dict['iteration'], cfg_dict['Task']
    date, TrainingFraction = cfg_dict['date'], int(cfg_dict['TrainingFraction'][0]*100)
    network_name = f"{Task}{date}-trainset{TrainingFraction}shuffle{shuffle}"

    pose_cfg_path = os.path.join(config_parent_dir, 
                                 "dlc-models",
                                 f"iteration-{iteration}",
                                 network_name,
                                 "train",
                                 "pose_cfg.yaml")
    
    with open(pose_cfg_path, 'r') as file:
        pose_cfg_dict = yaml.safe_load(file)
    
    return pose_cfg_dict, pose_cfg_path

def fetch_evaluation_csv_given_config(config_path : str):
    """
    Fetches the CombinedEvaluation-results.csv file path & content given 
    a config.yaml for a given DLC project. 
    CombinedEvaluation-results is located under:
    CONFIG_PARENT_DIR > evaluation-results > iteration-N > CombinedEvaluation-results.csv

    :param str config_path: Path to config.yaml of the DLC project.
    
    :returns csv_df, csv_path: csv read into a pandas.DataFrame, as well as its path.
    """
    # CONFIG_PARENT_DIR > evaluation-results > iteration-N > CombinedEvaluation-results.csv
    config_parent_dir = os.path.dirname(config_path)
    
    with open(config_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    iteration = cfg_dict['iteration']
    
    csv_path = os.path.join(config_parent_dir, 
                            "evaluation-results",
                            f"iteration-{iteration}",
                            "CombinedEvaluation-results.csv")
    csv_df = pd.read_csv(csv_path)
    return csv_df, csv_path


if __name__ == "__main__":
    # CONFIG_PATH = ""
    # SHUFFLE = 1
    # cfg_dict, cfg_path = fetch_pose_cfg_given_config(config_path=CONFIG_PATH, shuffle=SHUFFLE)
    # print("Pose config: ")
    # print(cfg_dict)
    # print("Pose config path: ")
    # print(cfg_path)

    # csv_df, csv_path = fetch_evaluation_csv_given_config(config_path=CONFIG_PATH)
    # print("Evaluation csv: ")
    # print(csv_df)
    # print("Evaluation csv path: ")
    # print(csv_path)

    CONFIG = r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/config.yaml"
    EARLYSTOP_NUM = 3

    from send_slack_message import send_slack_message

    start_message = f"Starting to train network with EARLYSTOP_NUM of {EARLYSTOP_NUM} and config found at: {CONFIG}."
    send_slack_message(message=start_message)

    try:
        train_with_early_stopping(
            config=CONFIG, # DLC original parameter
            earlystop_obs_num=EARLYSTOP_NUM,
            use_filtered_error=False,
            # DLC original parameters
            shuffle=1,
            max_snapshots_to_keep=None, 
            displayiters=1000, 
            saveiters=50000, 
            maxiters=3030000)
        
        success_message = "Training successfully completed!"
        send_slack_message(message=success_message)
    except Exception as e:
        print(e)
        send_slack_message(message="Training halted given unexpected error!")

    PLOT_SCORE_MAPS = False
    SHUFFLES = [1]
    COMPARISON_BODYPARTS = "all"
    
    send_slack_message(message="Starting evaluation!")
    
    try:
        deeplabcut.evaluate_network(CONFIG,
                                     Shuffles=SHUFFLES,
                                     plotting=PLOT_SCORE_MAPS,
                                     show_errors=True,
                                     comparisonbodyparts=COMPARISON_BODYPARTS
                                     )
        send_slack_message(message="Evaluation successfully completed!")
    except Exception as e:
        print(e)
        send_slack_message(message="Evaluation halted given unexpected error!")