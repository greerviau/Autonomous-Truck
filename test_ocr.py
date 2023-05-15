import numpy as np
import cv2 
import pytesseract as tess
from utils import get_speed, get_speed_limit
import time
import os


cap = cv2.VideoCapture('data/roof_cam/raw/sess_50/clip_4/drive.mp4')

from tensorflow.keras.models import load_model

ocr_model = load_model('digit_rec.h5')

data_dir = 'digit_data_2'


frame_count = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    speed, spd_1, spd_2 = get_speed(frame, ocr_model)

    speed_limit, spd_lim_1, spd_lim_2 = get_speed_limit(frame, ocr_model)


    print('Speed Limit: {} - Speed: {}'.format(speed_limit, speed))
    
    spd_str = str(speed)

    if not os.path.exists(os.path.join(data_dir, spd_str[0])):
        os.makedirs(os.path.join(data_dir, spd_str[0]))
        
    cv2.imwrite(data_dir+'/'+spd_str[0]+'/frame_{}.jpg'.format(frame_count), spd_1)

    frame_count += 1

    if not os.path.exists(os.path.join(data_dir, spd_str[1])):
        os.makedirs(os.path.join(data_dir, spd_str[1]))
        
    cv2.imwrite(data_dir+'/'+spd_str[1]+'/frame_{}.jpg'.format(frame_count), spd_2)

    frame_count += 1

    spd_lim_str = str(speed_limit)

    if not os.path.exists(os.path.join(data_dir, spd_lim_str[0])):
        os.makedirs(os.path.join(data_dir, spd_lim_str[0]))
        
    cv2.imwrite(data_dir+'/'+spd_lim_str[0]+'/frame_{}.jpg'.format(frame_count), spd_lim_1)

    frame_count += 1

    if not os.path.exists(os.path.join(data_dir, spd_lim_str[1])):
        os.makedirs(os.path.join(data_dir, spd_lim_str[1]))
        
    cv2.imwrite(data_dir+'/'+spd_lim_str[1]+'/frame_{}.jpg'.format(frame_count), spd_lim_2)

    frame_count += 1
    

    cv2.imshow('frame', frame)
    #time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
