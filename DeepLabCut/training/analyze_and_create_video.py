# Author: Akira Kudo
# Created: 2024/05/16
# Last Updated: 2024/05/28

import os

import deeplabcut

from .send_slack_message import send_slack_message

def analyze_and_create_video(config : str,
                             analyzed_videos : list,
                             rendered_videos : list,
                             shuffle : int,
                             filtered : bool,
                             save_frames : bool,
                             videotype : str="mp4",
                             destfolder : str=None, 
                             batchsize : int=8,
                             keypoints_only : bool=False,
                             displayedbodyparts : list="all",
                             outputframerate : int=None,
                             draw_skeleton : bool=False,
                             trailpoints : int=0):
    """
    Analyzes, then creates videos for specified videos (given as paths)
    with the specified settings.

    :param str config: Full path of the config.yaml file as a string.
    :param list analyzed_videos: A list of strings containing the full 
    paths to videos for analysis or a path to the directory, where all 
    the videos with same extension are stored.
    :param list rendered_videos: A list of strings containing the full 
    paths to videos for video generation or a path to the directory, 
    where all the videos with same extension are stored.
    :param int shuffle: Number of shuffles of training dataset.
    :param bool filtered: Boolean variable indicating if filtered output 
    should be plotted rather than frame-by-frame predictions. Filtered 
    version can be calculated with deeplabcut.filterpredictions
    :param bool save_frames: If true creates each frame individual and 
    then combines into a video. This variant is relatively slow as it 
    stores all individual frames.
    :param str videotype: Checks for the extension of the video in case 
    the input to the video is a directory. Only videos with this extension 
    are analyzed. Defaults to "mp4".
    :param str destfolder: Specifies the destination folder that was used 
    for storing analysis data (default is the path of the video).
    :param int batchsize: Change batch size for inference; if given, 
    overwrites value in ``pose_cfg.yaml``. Defaults to 8
    :param bool keypoints_only: By default, both video frames and keypoints 
    are visible. If true, only the keypoints are shown. Defaults to False.
    :param list displayedbodyparts: This selects the body parts that are 
    plotted in the video. Either ``all``, then all body parts from config.yaml 
    are used orr a list of strings that are a subset of the full list.
    Defaults to "all".
    :param int outputframerate: Output frame rate for labeled video 
    (only available for the mode with saving frames.) By default: None, 
    which results in the original video rate.
    :param bool draw_skeleton: If ``True`` adds a line connecting the body 
    parts making a skeleton on on each frame. The body parts to be connected 
    and the color of these connecting lines are specified in the config file. 
    By default: ``False``, defaults to False
    :param int trailpoints: Number of previous frames whose body parts are 
    plotted in a frame (for displaying history). Default is set to 0.
    """
    # first analyze the videos
    send_slack_message(message=f"Starting to analyze videos:\n" + 
                       " - " + "\n - ".join([os.path.basename(vid) for vid in analyzed_videos]))
    try:
        deeplabcut.pose_estimation_tensorflow.predict_videos.analyze_videos(
            config=config,
            videos=analyzed_videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=0,
            gputouse=0,
            save_as_csv=True,
            destfolder=destfolder,
            batchsize=batchsize,
            TFGPUinference=True
            )
    except Exception as e:
        print(e)
        send_slack_message(message="Video analysis unsuccessful...")
    
    # then create labeled videos from it
    send_slack_message(message=f"Starting to generate videos:\n" + 
                       " - " + "\n - ".join([os.path.basename(vid) for vid in rendered_videos]))
    try:
        deeplabcut.utils.make_labeled_video.create_labeled_video(
            config=config,
            videos=rendered_videos,
            videotype=videotype,
            shuffle=shuffle,
            trainingsetindex=0,
            filtered=filtered,
            fastmode=True,
            save_frames=save_frames,
            keypoints_only=keypoints_only,
            Frames2plot=None,
            displayedbodyparts=displayedbodyparts,
            outputframerate=outputframerate,
            destfolder=destfolder,
            draw_skeleton=draw_skeleton,
            trailpoints=trailpoints
        )
    except Exception as e:
        print(e)
        send_slack_message(message="Video generation unsuccessful...")

if __name__ == "__main__":
    CONFIG = os.path.join(r"/media/Data/Raymond Lab/YAC128-D2Cre Open Field/WhiteMice_Openfield-Ellen-2023-01-19",
                          # r"/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12",
                           "config.yaml")
    VIDEO_FOLDER = "/media/Data/Raymond Lab/YAC128 Mice Video Temp Akira Kudo"
    ANALYZED_VIDEOS = [os.path.join(VIDEO_FOLDER, file) for file in os.listdir(VIDEO_FOLDER)]
    RENDERED_VIDEOS = ANALYZED_VIDEOS

    SHUFFLE = 1
    FILTERED = False

    analyze_and_create_video(config=CONFIG,
                             analyzed_videos=ANALYZED_VIDEOS,
                             rendered_videos=RENDERED_VIDEOS,
                             shuffle=SHUFFLE,
                             filtered=FILTERED,
                             save_frames=False,
                             videotype="mp4",
                             destfolder=None, 
                             batchsize=8,
                             keypoints_only=False,
                             displayedbodyparts="all",
                             outputframerate=None,
                             draw_skeleton=False,
                             trailpoints=0)