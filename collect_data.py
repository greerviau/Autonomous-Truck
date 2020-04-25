import numpy as np
from PIL import ImageGrab
from xbox_controller import XboxController
from utils import grab_frame
import cv2
import time
import csv
import os
import sys


def main(collecting = True):

    camera_angle = 'roof_cam'

    xbox = XboxController()
    
    resolution = (1280, 720)
    fps = 30

    clip = 1

    data_dir = os.path.join('data', camera_angle, 'raw', sys.argv[1], 'clip_'+str(clip))

    if collecting:
        if os.path.exists(data_dir):
            print(data_dir, ' Already Exists!')
            return
        else:
            os.makedirs(data_dir)

        out_video = cv2.VideoWriter(os.path.join(data_dir, 'drive.mp4'),cv2.VideoWriter_fourcc(*'XVID'), fps, resolution)

        outputfile = open(os.path.join(data_dir, 'controls.csv'), 'w', newline='')
        csvwriter = csv.writer(outputfile)
        csvwriter.writerow(list(xbox.read().keys()))

    clipping = False
    try:
        while True:
            input_dict = xbox.read()
            #print(input_dict)

            print('Left X: {:.5f}'.format(input_dict['Left_X']))

            b_but = input_dict['B']

            if b_but == 1 and not clipping:
                clipping = True
                clip += 1
                out_video.release()

                data_dir = os.path.join('data', camera_angle, 'raw', sys.argv[1], 'clip_'+str(clip))
                os.makedirs(data_dir)

                out_video = cv2.VideoWriter(os.path.join(data_dir, 'drive.mp4'),cv2.VideoWriter_fourcc(*'XVID'), fps, resolution)

                outputfile = open(os.path.join(data_dir, 'controls.csv'), 'w', newline='')
                csvwriter = csv.writer(outputfile)
                csvwriter.writerow(list(xbox.read().keys()))

            elif b_but == 0:
                clipping = False

            frame = grab_frame(region = (0,0,1280,720))
            frame = cv2.resize(frame, resolution)
            cv2.imshow('window',cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if collecting:
                values = list(input_dict.values())
                #print(values)
                csvwriter.writerow(values)
                out_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

        print('Exiting')

main(collecting = True)