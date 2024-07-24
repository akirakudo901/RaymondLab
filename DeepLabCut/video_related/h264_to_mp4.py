import ffmpy
import os
import cv2

"""
** TO READ! IMPORTANT! **
 - The .h264 extension specifies that data (such as the video stream) contained in the file 
   was encoded using the H264 codec.
 - The .mp4 extension rather specifies that the file contains its different streams (video, audio, etc.)
   in the multimedia container format that is mp4.

Hence, the conversion of an h264 into a mp4 does not involve the decoding & re-encoding of the video stream, 
but rather a quick "re-packaging" of the data.
This can be done very quickly, and without loss of video quality, especially when time stamps are provided
together with the h264 file.

The above can be accomplished by specifying the ffmpeg command as follows:
1. ffmpeg -i INPUT_FILE -vcodec copy OUTPUT_FILE

In our case, our h264 seem to lack a time stamp file that must come with it (I have heard from
Marja that Ellen unfortunately didn't keep them). In such a case, running the above seems to 
create mp4 files with the default framerate of ffmpeg, 25.

In order to render the mp4 with a specific framerate, we need to specify it either with:
2. ffmpeg -r FRAMERATE -i INPUT_FILE -vcodec copy OUTPUT_FILE
or
3. ffmpeg -framerate FRAMERATE -i INPUT_FILE -vcodec copy OUTPUT_FILE

The difference of which I do not completely understand, but the ffmpeg documentation suggests
to use '-framerate' when in doubt (see here[https://ffmpeg.org/ffmpeg.html#Video-Options], 2nd entry).

According to Tony from Tim's lab, however, using 2. or 3. might result in a loss of video quality.
He hence suggested to use 1. instead.
Although I did not verify whether using 2. and 3. indeed leads to a loss compared to 1., 
proceed with above points in mind when using this script!
"""

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
        # according to 
        inputs={f'{video_path}': f' -r {framerate}'},
        outputs={f'{video_path_mp4}':'-vcodec copy'}
        )
    
    print(ff.cmd)
    raise Exception()
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
    VIDEO_FOLDER_PATH = r"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Videos (not organized)\Open Field - Brown Mice for Behaviour Analysis Training"
    FRAMERATE = 40

    for file_path in [os.path.join(VIDEO_FOLDER_PATH, file) for file in os.listdir(VIDEO_FOLDER_PATH)]:
        n_frames = convert2mp4(file_path, FRAMERATE)
        # check if the number of frames is correct
        print(n_frames)