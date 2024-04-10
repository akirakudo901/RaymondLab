# Author: Akira Kudo
# Created: 2024/04/07
# Last Updated: 2024/04/10

import os
from typing import List

import ffmpy

def extract_frame_by_number(input_file : str, 
                            output_file : str, 
                            frame_numbers : List[int]):
    """
    Extracts from an mp4 input_file, the frames specified by frame_numbers, and
    save them under the name of output_file with a number appended at the end
    indicating which frame the output file corresponds to.

    :param str input_file: The path to the input mp4 video.
    :param str output_file: The name every extracted frames will have (including
    its extension), to which we append its frame number. e.g. OUTPUT3.jpg
    :param List[int] frame_numbers: A list of integer indicating which frame to 
    extract.
    """
    # specify selected frames
    select_str = "select='"
    select_str += '+'.join(['eq(n\,{})'.format(fr_num) for fr_num in frame_numbers])
    select_str += "'"
    # output file is named by the nth output from the pipeline
    img_extension = os.path.splitext(output_file)[-1]
    nth_output_file = output_file.replace('{}'.format(img_extension), 
                                          r'%d{}'.format(img_extension))
    
    # Create FFmpeg command to extract frame by frame number
    ff = ffmpy.FFmpeg(
        inputs={input_file : None},
        outputs={nth_output_file : f'-vf "{select_str}" -vsync vfr'}
    )
    
    # Execute the FFmpeg command
    ff.run()

    # rename the resulting frames by their position in the video
    for img_idx in range(1, len(frame_numbers) + 1):
        img_idx_th_output_file = output_file.replace(f'{img_extension}', 
                                                     f'{img_idx}{img_extension}')
        frame_number_output_file = output_file.replace(f'{img_extension}', 
                                                       f'{frame_numbers[img_idx - 1]}{img_extension}')
        os.rename(img_idx_th_output_file, frame_number_output_file)

if __name__ == "__main__":
    # Specify input MP4 file, output PNG file, and frame number of the frame to extract
    input_folder = '/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/videos'
    input_name = '20220228231804_320151_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered_labeled.mp4'
    
    output_folder = '/media/Data/Raymond Lab/to_delete_Python_Scripts/video_related/data/extracted_frames'
    output_name = 'extracted.png'
    
    
    input_file = os.path.join(input_folder, input_name)
    output_file = os.path.join(output_folder, output_name)
    
    frame_numbers = list(range(50, 61))
    
    # Extract the frame
    extract_frame_by_number(input_file, output_file, frame_numbers)
