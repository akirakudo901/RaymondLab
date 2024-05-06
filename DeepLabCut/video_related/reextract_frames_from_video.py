# Author: Akira Kudo
# Created: 2024/04/25
# Last Updated: 2024/04/25

import os

from video_related.extract_image_from_mp4 import extract_frame_by_number

def reextract_frames_from_video(img_dir_path : str, 
                                video_path : str,
                                outdir_path : str):
    """
    Reextarcts the frames which are contained in img_dir_path 
    under the name 'img[NUMBER].png' from specified video, 
    outputting them with the same name into outdir_path.

    :param str img_dir_path: Path to directory holding images.
    :param str video_path: Path to video frames are extracted from. 
    :param str outdir_path: Path to which extracted frames are outputted.
    """
    # determine which frames to extract
    frames_indices_to_extract = []
    for file in os.listdir(img_dir_path):
        if file.startswith('img') and file.endswith('.png'):
            file_index = int(file.replace('img', '').replace('.png', ''))
            frames_indices_to_extract.append(file_index)
    
    # extract these frames and output them into outdir_path
    extract_frame_by_number(input_file=video_path, 
                            output_file=os.path.join(outdir_path, 'img.png'), 
                            frame_numbers=frames_indices_to_extract,
                            scaling_factor=1.0)
    
    # rename the extracted frames by the correct naming convention
    for file in os.listdir(outdir_path):
        if file.startswith('img') and file.endswith('.png'):
            file_index = int(file.replace('img', '').replace('.png', ''))
            padded_name = 'img{:05d}.png'.format(file_index)
            os.rename(os.path.join(outdir_path, file),
                      os.path.join(outdir_path, padded_name))