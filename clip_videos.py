import numpy as np
import pandas as pd
import cv2
import os
import sys
import time

def clipper(d_dir):
    path = os.path.join(data_dir,d_dir)
    
    video = cv2.VideoCapture(path+'/drive.mp4')
    
    split_at = [0]
    c = 0
    while True:

        ret, frame = video.read()

        if not ret:
            break

        c += 1
        
        cv2.imshow('Clipper', frame)
        time.sleep(0.01)

        if c % 1 == 0:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Key Frame: ',c)
                split_at.append(c)

    split_at.append(c)
    video.release()
    cv2.destroyAllWindows()
    return (d_dir, split_at)

def process_clips(clip_data):
    d_dir = clip_data[0]
    clip_at = clip_data[1]

    path = os.path.join(data_dir,d_dir)

    processed_path = os.path.join(cleaned_data_dir, d_dir)

    video = cv2.VideoCapture(path+'/drive.mp4')
    data = pd.read_csv(path+'/controls.csv')
    c = 0

    for clip in range(1,len(clip_at)):
        clip_dir = os.path.join(processed_path, 'split_'+str(clip))
        if not os.path.exists(clip_dir):
            os.makedirs(clip_dir)
        out_video = cv2.VideoWriter(clip_dir+'/drive.mp4',cv2.VideoWriter_fourcc(*'XVID'), 30, resolution)
        while True:
            ret, frame = video.read()
            if not ret:
                break

            out_video.write(cv2.resize(frame, resolution))

            c+= 1

            if c == clip_at[clip]:
                out_video.release()
                print(clip_at[clip-1],clip_at[clip])
                data.iloc[clip_at[clip-1]:clip_at[clip],:].to_csv(clip_dir+'/controls.csv')
                break
        out_video.release()
    video.release()

if __name__ == '__main__':

    data_dir = 'data/roof_cam'

    resolution = (1280,720)

    cleaned_data_dir = os.path.join(data_dir, 'cleaned')

    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)

    data_folders = os.listdir(data_dir)

    if(len(sys.argv) > 1):
        split_info = clipper(sys.argv[1])
        process_clips(split_info)
    else:
        split_info = []
        for i, folder in enumerate(data_folders, 0):
            print('Clip {}/{}'.format(i+1, len(data_folders)))
            split_info.append(clipper(folder))

        for split_i in split_info:
            process_clips((split_i))
        
            
    
    
        