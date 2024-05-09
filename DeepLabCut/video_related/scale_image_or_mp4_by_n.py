# Author: Akira Kudo
# Created: 2024/03/16
# Last updated: 2024/04/24

import os

import ffmpy

def scale_image_by_n(img_path : str, n : float, outdir : str, newname : str=None, overwrite : bool=False):
    """
    Scales a given image file by n, giving it a new name and writing it to 'outdir'.
    :param str img_path: Path to image to be scaled, png/jpg.
    :param float n: Scaling factor.
    :param str outdir: The output directory to which the scaled image is output.
    :param str newname: The new name given to the scaled image. If not given with 
    extension, extension of the input is used. If not given at all, a name is given.
    :param bool overwrite: Overwrite when target output file exists, defaults to False.
    """
    if not img_path.endswith('.png') and not img_path.endswith('.jpg') and \
       not img_path.endswith('.jpeg'): 
        raise Exception("img_path doesn't seem to specify a png/jpg file...")
    
    _scale_something_by_n_using_ffmpy(obj_path=img_path, n=n, outdir=outdir, newname=newname, overwrite=overwrite)

def scale_mp4_by_n(vid_path : str, n : float, outdir : str, newname : str=None, overwrite : bool=False):
    """
    Scales a given image file by n, giving it a new name and writing it to 'outdir'.
    :param str vid_path: Path to video to be scaled, mp4.
    :param float n: Scaling factor.
    :param str outdir: The output directory to which the scaled video is output.
    :param str newname: The new name given to the scaled video. If not given with 
    extension, extension of the input is used. If not given at all, a name is given.
    :param bool overwrite: Overwrite when target output file exists, defaults to False.
    """
    if not vid_path.endswith('.mp4'):
        raise Exception("vid_path doesn't seem to specify an mp4 file...")
    
    _scale_something_by_n_using_ffmpy(obj_path=vid_path, n=n, outdir=outdir, newname=newname, overwrite=overwrite)


# HELPER
def _scale_something_by_n_using_ffmpy(obj_path : str, 
                                      n : float, 
                                      outdir : str, 
                                      newname : str=None,
                                      overwrite : bool=False):
    """
    Scales a given image / mp4 file by n, giving it a new name and writing it to 'outdir'.
    :param str obj_path: Path to object to be scaled, png/jpg/mp4.
    :param float n: Scaling factor.
    :param str outdir: The output directory to which the scaled object is output.
    :param str newname: The new name given to the scaled object. If not given with 
    extension, extension of the input is used. If not given at all, a name is given.
    :param bool overwrite: Overwrite when target output file exists, defaults to False.
    """
    
    # manipulate names as needed
    split_by_dot = os.path.basename(obj_path).split('.')
    filename_no_ext, ext = '.'.join(split_by_dot[:-1]), split_by_dot[-1]
    # if newname isn't given, create it too
    if newname is None:
        n_but_filesafe = str(n).replace('.', 'point')
        newname = f'{filename_no_ext}_rescaled_by_{n_but_filesafe}.{ext}'
    elif '.' not in newname: # append extension as needed
        newname = f'{newname}.{ext}'
        
    if not overwrite and os.path.exists(os.path.join(outdir, newname)):
        print(f"{newname} already exists in {outdir}!")
        return
    
    output_comm = f'-vf "scale=iw*{n}:ih*{n}"' + \
                  f' {"-c:v libx264 -crf 0 -preset veryslow -c:a copy" if ext == "mp4" else ""}'

    ff = ffmpy.FFmpeg(
             inputs = {f'{obj_path}' : None},
             outputs= {f'{os.path.join(outdir, newname)}' : output_comm}
             )
    # product of command should look like:
    #   ffmpeg -i input.png -vf "scale=iw/2:ih/2" input_half_size.png
    # for images, and 
    #   ffmpeg -i input.mp4 -vf "scale=iw*2:ih*2" -c:a copy input_rescaled_by2.mp4
    # for mp4
    
    ff.run()

if __name__ == "__main__":
    N1 = 0.5; N2 = 2

    IMG_TO_SCALE = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\BSOID\COMPUTED\results\pngs\320151m1\frame27.png"
    VIDEO_TO_SCALE = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\video_related\data\group_6_top1.mp4"

    OUTDIR = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\video_related\data\result"
    
    scale_image_by_n(img_path=IMG_TO_SCALE, 
                     n=N1, 
                     outdir=OUTDIR, 
                     newname=None)
    
    scale_image_by_n(img_path=IMG_TO_SCALE, 
                     n=N2, 
                     outdir=OUTDIR, 
                     newname=None)
    
    scale_mp4_by_n(vid_path=VIDEO_TO_SCALE, 
                   n=N1, 
                   outdir=OUTDIR,
                   newname=None)
    
    scale_mp4_by_n(vid_path=VIDEO_TO_SCALE, 
                   n=N2, 
                   outdir=OUTDIR,
                   newname=None)