# Author: Akira Kudo
# Created: 2024/03/12
# Last updated: 2024/03/16

import os

import ffmpy
import tqdm

def scale_mp4_by_n(mp4_path, n, outdir=None):
    """
    Scales an mp4 file by n.
    Originally to downsample files from 1080x1080 to 540x540 to save space.
    """
    if not os.path.exists(mp4_path): raise Exception("mp4_path could not be found...")
    elif not os.path.isfile(mp4_path): raise Exception("mp4_path isn't a file...")
    elif not mp4_path.endswith('.mp4'): raise Exception("mp4_path doesn't seem to specify an mp4 file...")

    # if outdir isn't specified, output to the same place as the video comes
    if outdir is None: outdir = os.path.dirname(mp4_path)
    
    current_path = os.getcwd()
    
    dir_path = os.path.dirname(mp4_path)
    os.chdir(dir_path)
    
    filename = os.path.basename(mp4_path).replace('.mp4', '')
    n_but_filesafe = str(n).replace('.', 'point')
    scaled_filename = filename + f'_rescaled_by_{n_but_filesafe}'
    
    if os.path.exists(os.path.join(outdir, scaled_filename)):
        print(f"{scaled_filename} already exists in {outdir}!")
        os.chdir(current_path)
        return
    
    ff = ffmpy.FFmpeg(
             inputs = {f'{filename}.mp4' : None},
             outputs= {f'{scaled_filename}.mp4' : f'-vf "scale=iw*{n}:ih*{n}" -c:a copy'}
             )
    # if n is 2, for example, product of command should look like:
    # ffmpeg -i input.mp4 -vf "scale=iw*2:ih*2" -c:a copy input_rescaled_by2.mp4
    ff.run()
    
    # move the result to the specified outdir
    os.rename(os.path.join(dir_path, f"{scaled_filename}.mp4"),
              os.path.join(outdir,   f"{scaled_filename}.mp4"))

    os.chdir(current_path)

if __name__ == "__main__":
    MP4_FOLDER = "/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/videos/temp"
# "/media/Data/Raymond Lab/Python_Scripts/video_related/data"
    # MP4_FILE = "group_6_top1.mp4"
    # mp4_path = os.path.join(MP4_FOLDER, MP4_FILE)
    
    OUTDIR = "/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Akira-2024-03-12 Q175-D2Cre Open Field Males Brown/videos"
    
    for file in tqdm.tqdm(os.listdir(MP4_FOLDER)):
        mp4_path = os.path.join(MP4_FOLDER, file)
        try:
            scale_mp4_by_n(mp4_path, n=0.5, outdir=OUTDIR)
        except Exception as e:
            print(e)
    
