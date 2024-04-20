import ffmpy
import os
import cv2

# To execute in Anaconda Prompt:
# 1. Change VIDEO_PATHS to the list of paths to h246 we wanna convert
# 2. Enter:      conda activate mpi_analysis_gpu
# 3. Enter:      python PATH_TO_THIS_FILE 
#    replacing PATH_TO_THIS_FILE with the actual path to this file. E.g.:
#    python "Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\video_related\h264_to_mp4.py"
 
def convert2mp4(video_path : str, framerate : int):
    """
    Converts a given h264 file to mp4.

    :param str video_path: Path to the video to convert.
    :param int framerate: Framerate of the created video.
    """
    videoname = os.path.basename(video_path)
    video_path_mp4 = video_path.replace('.h264', '.mp4')
    # double check vid is a h264 file
    if not videoname.endswith('.h264'):
        raise Exception(f"The passed file path {videoname} needs to be a h264...")
    
    # execute the ffmpy command
    ff = ffmpy.FFmpeg(
        inputs={f'{video_path}': f' -r {framerate}'},
        outputs={f'{video_path_mp4}':'-vcodec copy'}
        )
    try:
        ff.run()
        print('mp4 file generated')
    except Exception as e:
        print(e)
        print('please check folder path')

    video = cv2.VideoCapture(f'{video_path_mp4}')
    vid_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    
    return vid_length


if __name__ == "__main__":
    VIDEO_FOLDER_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira_newQ175_to_convert"
    FRAMERATE = 40

    for file_path in [os.path.join(VIDEO_FOLDER_PATH, file) for file in os.listdir(VIDEO_FOLDER_PATH)]:
        n_frames = convert2mp4(file_path, FRAMERATE)
        # check if the number of frames is correct
        print(n_frames)



