# Author: Akira Kudo
# Created: 2024/03/16
# Last updated: 2024/03/16

import os

import ffmpy
import tqdm

def scale_image_by_n(img_path, n, newname : str, outdir=None):
    """
    Scales an image file by n.
    """
    if not os.path.exists(img_path): raise Exception("img_path could not be found...")
    elif not os.path.isfile(img_path): raise Exception("img_path isn't a file...")
    elif not img_path.endswith('.png'): raise Exception("img_path doesn't seem to specify a png file...")

    # if outdir isn't specified, output to the same place as the image comes
    if outdir is None: outdir = os.path.dirname(img_path)
    
    current_path = os.getcwd()
    
    dir_path = os.path.dirname(img_path)
    os.chdir(dir_path)
    
    filename = os.path.basename(img_path).replace('.png', '')
    n_but_filesafe = str(n).replace('.', 'point')
    scaled_filename = filename + f'_rescaled_by_{n_but_filesafe}'
    
    if os.path.exists(os.path.join(outdir, scaled_filename)):
        print(f"{scaled_filename} already exists in {outdir}!")
        os.chdir(current_path) 
        return
    
    ff = ffmpy.FFmpeg(
             inputs = {f'{filename}.png' : None},
             outputs= {f'{scaled_filename}.png' : f'-vf "scale=iw*{n}:ih*{n}"'}
             )
    # product of command should look like:
    # ffmpeg -i input.png -vf "scale=iw/2:ih/2" input_half_size.png
    ff.run()
    
    os.chdir(current_path)
    
    # move the result to the specified outdir
    if newname.endswith('.png'): newname = newname.replace('.png', '')
    os.rename(os.path.join(dir_path, f"{scaled_filename}.png"),
              os.path.join(outdir,   f"{newname}.png"))

if __name__ == "__main__":
    N = 0.5
    n_filesafe = str(N).replace('.', 'point')
    ABOVE_IMG_FOLDER = "/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/labeled-data"
    IMG_FOLDER = "20230107131118_363453_m1_openfield"
    #"20230107123308_362816_m1_openfield"
    #"20230102092905_363451_f1_openfield"
    #"20220211070325_301533_f3_"
    
    IMG_DIR = os.path.join(ABOVE_IMG_FOLDER, IMG_FOLDER)
    
    OUTDIR = os.path.join("/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown Halfscale-Akira-2024-03-15/labeled-data", IMG_FOLDER + f"_rescaled_by_{n_filesafe}")
    
    for filename in tqdm.tqdm(os.listdir(IMG_DIR)):
        img_path = os.path.join(IMG_DIR, filename)
        try:
            scale_image_by_n(img_path, n=N, outdir=OUTDIR, newname=filename)
        except Exception as e:
            print(e)
    
