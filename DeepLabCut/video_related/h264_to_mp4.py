import ffmpy
import os
import cv2
# Change the path below to the folder that your h264 files are in
# In terminal, enter cd /media/Data/'Raymond Lab'  [or whatever directory this file is in]
# Then enter conda activate mpi_analysis_gpu
# Then enter ipython h264_to_mp4.py 
 
def convert2mp4(vid,framerate):
    current_path = os.getcwd()
    #vid=vid_path +'/raw.h264'
    path = os.path.dirname(vid)
    bname = os.path.basename(vid)[:-5]
    os.chdir(path)
    ff = ffmpy.FFmpeg(inputs={f'{bname}.h264': f' -r {framerate} -i {bname}.h264'},outputs={f'{bname}.mp4':'-vcodec copy'})
    try:
        ff.run()
        print('mp4 file generated')
    except Exception as e:
        print(e)
        print('please check folder path')
    vid = cv2.VideoCapture(f'{bname}.mp4')
    vid_length=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release()
    os.chdir(current_path)
    return vid_length

#path='/media/Data/Raymond Lab/YAC128-D2Cre Open Field/Open Field Videos/Females/YAC128/' # enter path here end with /
framerate=40
path='/media/Data/Raymond Lab/Q175-D2Cre Open Field Males/Q175-D2Cre Open Field Males Brown-Judy-2024-01-12/videos/temp_for_conversion/'



files = [path+i for i in os.listdir(path)]
for file in files:
    n_frames=convert2mp4(file,framerate)
    print(n_frames)# probably want to save this just to check if the number of frames is correct



