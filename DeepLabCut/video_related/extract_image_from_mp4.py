# Author: Akira Kudo
# Created: 2024/04/07
# Last Updated: 2024/04/12

import itertools
import os
from typing import List

import ffmpy

FPS_FOR_FRAMENUMBER_TO_SECONDS_CONVERSION = 40

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
    # below code thanks to: 
    # https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python/4629241#4629241
    def to_ranges(iterable):
        iterable = sorted(set(iterable))
        for _, group in itertools.groupby(enumerate(iterable),
                                            lambda t: t[1] - t[0]):
            group = list(group)
            yield group[0][1], group[-1][1]
    
    def frame2seconds(frame_number : int):
        return frame_number / FPS_FOR_FRAMENUMBER_TO_SECONDS_CONVERSION

    # first aggregate the frame numbers using to_ranges, then iterate through
    for start_frame, end_frame in to_ranges(frame_numbers):
        # output file is named by the nth output from the pipeline
        img_extension = os.path.splitext(output_file)[-1]
        nth_output_file = output_file.replace('{}'.format(img_extension), 
                                            r'TEMP%d{}'.format(img_extension))
        
        # Create FFmpeg command to extract frame by frame number
        ff = ffmpy.FFmpeg(
            inputs={input_file : f' -ss {frame2seconds(start_frame)}'},
            # we wanna add -frames:v so that the 
            outputs={nth_output_file : f'-vsync vfr -frames:v {end_frame-start_frame+1}'}
        )
        
        # Execute the FFmpeg command
        ff.run()

        # rename the resulting frames by their position in the video
        for temp_idx, frame_num in enumerate(range(start_frame, end_frame+1)):
            temp_idx += 1 # temporal indices are 1-indexed
            temp_idx_th_output_file = output_file.replace(f'{img_extension}', 
                                                          f'TEMP{temp_idx}{img_extension}')
            frame_number_output_file = output_file.replace(f'{img_extension}', 
                                                        f'{frame_num}{img_extension}')
            if os.path.exists(frame_number_output_file):
                os.remove(temp_idx_th_output_file)
            else:
                os.rename(temp_idx_th_output_file, frame_number_output_file)

if __name__ == "__main__":
    # Specify input MP4 file, output PNG file, and frame number of the frame to extract
    # input_folder = '/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/videos'
    # input_name = '20220228231804_320151_m2_openfieldDLC_resnet50_Q175-D2Cre Open Field Males BrownJan12shuffle1_1030000_filtered_labeled.mp4'
    
    # output_folder = '/media/Data/Raymond Lab/to_delete_Python_Scripts/video_related/data/extracted_frames'
    # output_name = 'extracted.png'    
    
    # input_file = os.path.join(input_folder, input_name)
    # output_file = os.path.join(output_folder, output_name)
    
    # frame_numbers = list(range(50, 61))
    
    # # Extract the frame
    # extract_frame_by_number(input_file, output_file, frame_numbers)

    inputfile = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC_data\Q175\videos\generated\320151m1_0to1000_20fps.mp4"
    outputfile= r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC_data\Q175\videos\labeled_extracted\extracted.jpg"
    extract_frame_by_number(inputfile, outputfile, list(range(1000)))