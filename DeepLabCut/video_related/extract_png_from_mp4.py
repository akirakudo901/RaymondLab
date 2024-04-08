# Author: Akira Kudo
# Created: 2024/04/07
# Last Updated: 2024/04/07

import os

import ffmpy

def extract_frame_by_number(input_file, output_file, frame_numbers : list):
    # specify selected frames
    select_str = "select='"
    select_str += '+'.join(['eq(n\,{})'.format(fr_num) for fr_num in frame_numbers])
    select_str += "'"
    # output file is named by the nth output from the pipeline
    nth_output_file = output_file.replace('.png', r'%d.png')
    
    # Create FFmpeg command to extract frame by frame number
    ff = ffmpy.FFmpeg(
        inputs={input_file : None},
        outputs={nth_output_file : f'-vf "{select_str}" -vsync vfr'}
    )
    
    print(ff.cmd)
    
    # Execute the FFmpeg command
    ff.run()

    # rename the resulting frames by their position in the video
    for img_idx in range(1, len(frame_numbers) + 1):
        img_idx_th_output_file = output_file.replace('.png', f'{img_idx}.png')
        frame_number_output_file = output_file.replace('.png', f'{frame_numbers[img_idx - 1]}.png')
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
