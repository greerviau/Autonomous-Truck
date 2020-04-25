import cv2
import time
import csv
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import ImageGrab
from xbox_controller import XboxController
from utils import grab_frame
from conv_net_model import *

from tensorflow.keras.models import load_model

from utils import get_speed, get_speed_limit, get_curve
from direct_keys import PressKey, ReleaseKey, W, A, S, D, L

import pyvjoy

MAX_VJOY = 32767
resolution = (1280, 720)
max_speed = 50

def set_joy_from_gamepad(joy, pad_dict):

    joy.set_button(1, pad_dict['A']* MAX_VJOY)
    joy.set_button(2, pad_dict['B']* MAX_VJOY)
    joy.set_button(3, pad_dict['X']* MAX_VJOY)
    joy.set_button(4, pad_dict['Y']* MAX_VJOY)
    joy.set_button(5, pad_dict['LB']* MAX_VJOY)
    joy.set_button(6, pad_dict['RB']* MAX_VJOY)
    joy.set_button(7, pad_dict['Back']* MAX_VJOY)
    joy.set_button(8, pad_dict['Start']* MAX_VJOY)
    joy.set_button(9, pad_dict['LStick']* MAX_VJOY)
    joy.set_button(10, pad_dict['RStick']* MAX_VJOY)

    #joy.data.wAxisZ = int(pad_dict['LT'] * MAX_VJOY)

    #joy.data.wAxisZRot = int(pad_dict['RT'] * MAX_VJOY)
    
    #joy.data.wAxisZ = int((0.5 - (pad_dict['LT']/2) + (pad_dict['RT']/2)) * MAX_VJOY)

    #joy.data.wAxisZRot = int((0.5 - (pad_dict['LT']/2) + (pad_dict['RT']/2)) * MAX_VJOY)
    
    #joy.data.wAxisXRot = int(((1.0+pad_dict['Right_X'])/2) * MAX_VJOY)
    #joy.data.wAxisYRot = MAX_VJOY - int(((1.0+pad_dict['Right_Y'])/2) * MAX_VJOY)


def main():

    if not os.path.exists('test_videos'):
        os.makedirs('test_videos')

    #video_out = cv2.VideoWriter("test_videos/test_3.mp4",cv2.VideoWriter_fourcc(*'XVID'), 30,(1280,720))

    network = conv_net(x, keep_prob)
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, 'conv_net_models/conv_net_v15_aug_bright_Peterbilt/conv_net.ckpt')

    xbox = XboxController()

    joy = pyvjoy.VJoyDevice(1)

    ocr_model = load_model('digit_rec_model/digit_rec.h5')

    #brake_net = load_model('brake_net_model/brake_net.h5')

    human_override=True

    lock = False

    lights_on = False

    light_avg = []

    inference_time = 1

    clip = 1
    collecting_data = False
    
    if collecting_data:
        data_dir = os.path.join('data/roof_cam/raw_autonomous', sys.argv[1], 'clip_'+str(clip))

        if os.path.exists(data_dir):
            print(data_dir, ' Already Exists!')
            return
        else:
            os.makedirs(data_dir)

        video_data = cv2.VideoWriter(os.path.join(data_dir, 'drive.mp4'),cv2.VideoWriter_fourcc(*'XVID'), 30, resolution)

        outputfile = open(os.path.join(data_dir, 'controls.csv'), 'w', newline='')
        csvwriter = csv.writer(outputfile)
        csvwriter.writerow(['Left_X', 'LT'])

    try:

        frame_count = 0
        speed_limit = 0

        while True:

            start = time.time()

            input_dict = xbox.read()

            joy.reset()

            set_joy_from_gamepad(joy, input_dict)

            left_x = input_dict['Left_X']
            left_y = input_dict['Left_Y']

            left_b = input_dict['LB']

            if left_b == 1 and not lock:
                if human_override:
                    human_override=False

                    if collecting_data:
                        data_dir = os.path.join('data/roof_cam/raw_autonomous', sys.argv[1], 'clip_'+str(clip))
                        if not os.path.exists(data_dir):
                            os.makedirs(data_dir)

                        video_data = cv2.VideoWriter(os.path.join(data_dir, 'drive.mp4'),cv2.VideoWriter_fourcc(*'XVID'), 30, resolution)

                        outputfile = open(os.path.join(data_dir, 'controls.csv'), 'w', newline='')
                        csvwriter = csv.writer(outputfile)
                        csvwriter.writerow(['Left_X', 'LT'])

                        clip += 1

                elif not human_override:
                    ReleaseKey(W)
                    ReleaseKey(S)
                    human_override=True
                    
                    video_data.release()
                
                lock = True

            if left_b == 0 and lock:
                lock = False


            frame = grab_frame(region = (0,0,1280,720))
            frame = cv2.resize(frame, resolution)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if not human_override:
                video_data.write(frame)

            light_avg.append(np.mean(frame[0,:,:]))
            if len(light_avg) > 50:
                light_avg.pop(0)

            light = np.mean(light_avg)

            road = frame[380:630, 330:950, :]  #W, H = 620, 250
            road = cv2.resize(road, (124, 50))

            input_frame = np.copy(road).astype(np.float16)

            pred_left_x = sess.run(network, feed_dict={x:[input_frame], keep_prob:1.0})[0][0]

            human_joy_x = int(((1.0+left_x)/2) * MAX_VJOY)

            ai_joy_x = int(((1.0+pred_left_x)/2) * MAX_VJOY)

            joy.data.wAxisY = MAX_VJOY // 2
            #joy.data.wAxisY = MAX_VJOY - int(((1.0+left_y)/2) * MAX_VJOY)

            joy.update()

            font = cv2.FONT_HERSHEY_SIMPLEX

            #BRAKE 
            #brake_pred = brake_net.predict(np.array([input_frame]))[0][0]
            #brake = brake_pred > 0.85
            brake_pred = 0
            brake = False

            #COLLECT DATA
            
            if not human_override and collecting_data:
                csvwriter.writerow([pred_left_x, brake_pred])
            
            #GET SPEEDS

            speed = get_speed(frame, ocr_model)

            if frame_count % 50 == 0:
                speed_limit = get_speed_limit(frame, ocr_model)
                frame_count = 0

            frame_count+=1

            if not human_override:

                joy.data.wAxisX = ai_joy_x

                if light < 80 and not lights_on:
                    PressKey(L)
                    ReleaseKey(L)
                    time.sleep(0.01)
                    PressKey(L)
                    ReleaseKey(L)
                    lights_on = True
                elif light > 80 and lights_on:
                    PressKey(L)
                    ReleaseKey(L)
                    lights_on = False

                if brake or speed - min(speed_limit, max_speed) > 5:
                    ReleaseKey(W)
                    PressKey(S)
                    brake = True
                elif speed < min(speed_limit, max_speed):
                    ReleaseKey(S)
                    PressKey(W)
                else:
                    ReleaseKey(S)
                    ReleaseKey(W)
                
            else:
                joy.data.wAxisX = human_joy_x


            auto_pilot_status = 'Disengaged'
            acc_status = 'Inactive'
            color = (0,0,255)
            if not human_override:
                auto_pilot_status = 'Engaged'
                acc_status = 'Active'
                color=(0,255,0)

            headlights = 'Off'
            headlight_color = (0,0,255)
            if lights_on:
                headlights = 'On'
                headlight_color = (0,255,0)


            def draw_text_to_frame(frame):
                cv2.putText(frame, 'Autopilot: ', (20,350), font, 0.8, (255,255,255), 2)
                #cv2.putText(frame, 'Autopilot: ', (20,350), font, 0.8, (0,0,0), 2)
                cv2.putText(frame, auto_pilot_status, (150,350), font, 0.8, color, 2)

                cv2.putText(frame, 'Predicted Turn: {:.3f}'.format(pred_left_x), (20,380), font, 0.8, (255,255,255), 2)
                #cv2.putText(frame, 'Predicted Turn: {:.3f}'.format(pred_left_x), (20,380), font, 0.8, (0,0,0), 2)

                cv2.putText(frame, 'Joy X: '+str(joy.data.wAxisX), (20,410), font, 0.8, (255,255,255), 2)
                #cv2.putText(frame, 'Joy X: '+str(joy.data.wAxisX), (20,410), font, 0.8, (0,0,0), 2)

                cv2.putText(frame, 'Auto CC: ', (20,460), font, 0.8, (255,255,255), 2)
                #cv2.putText(frame, 'Auto CC: ', (20,460), font, 0.8, (0,0,0), 2)
                cv2.putText(frame, acc_status, (140,460), font, 0.8, color, 2)

                cv2.putText(frame, 'Speed Limit: {}'.format(speed_limit), (20,490), font, 0.8, (255,255,255), 2)
                #cv2.putText(frame, 'Speed Limit: {}'.format(speed_limit), (20,490), font, 0.8, (0,0,0), 2)

                cv2.putText(frame, 'Max Speed: {}'.format(max_speed), (20,520), font, 0.8, (255,255,255), 2)
                #cv2.putText(frame, 'Max Speed: {}'.format(max_speed), (20,520), font, 0.8, (0,0,0), 2)

                cv2.putText(frame, 'Speed: {}'.format(speed), (20,550), font, 0.8, (255,255,255), 2)
                #cv2.putText(frame, 'Speed: {}'.format(speed), (20,550), font, 0.8, (0,0,0), 2)

                if not brake:
                    cv2.putText(frame, 'GAS ', (20,580), font, 0.8, (0,255,0), 2)
                else:
                    cv2.putText(frame, 'BRAKE ', (20,580), font, 0.8, (0,0,255), 2)

                cv2.putText(frame, 'Headlights: ', (20,630), font, 0.8, (255,255,255), 2)
                #cv2.putText(frame, 'Headlights: ', (20,630), font, 0.8, (0,0,0), 2)
                cv2.putText(frame, headlights, (170,630), font, 0.8, headlight_color, 2)

                cv2.putText(frame, 'Inf Time: {:.2f} fps'.format(1.0/inference_time), (20,660), font, 0.8, (255,255,255), 2)

                return frame

            shp = road.shape
            road = cv2.resize(road, (shp[1]*2, shp[0]*2))
            shp = road.shape
            
            overlay = np.copy(frame)
            cv2.rectangle(overlay, (0,320), (330,680), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 1-0.6, 1)
            frame = draw_text_to_frame(frame)

            frame[0:shp[0], 640-shp[1]//2:640+shp[1]//2, :] = road
            cv2.imshow('Auto Truck',frame)
            #video_out.write(frame)
            
            ''' 
            black_screen = np.zeros_like(frame)
            black_screen[0:shp[0], 640-shp[1]//2:640+shp[1]//2, :] = cv2.cvtColor(road, cv2.COLOR_BGR2RGB)

            black_screen = draw_text_to_frame(black_screen)
            cv2.imshow('Auto Truck Black',black_screen)
            '''

            '''
            curve = get_curve(pred_left_x*-1, frame.shape[1]//2, frame.shape[0])

            color = (0,0,255)
            if not human_override:
                color = (0,255,0)
            cv2.polylines(frame,[curve],False,color,10)
            '''

            #print('\rTruth: {:.5f} - Predicted: {:.5f} - True Joy X: {} - Pred Joy X: {} - Light Level: {:.2f} - Lights On: {} - AI Control: {}   '.format(left_x, pred_left_x, human_joy_x, ai_joy_x, light, str(lights_on), str(not human_override)), end='')


            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

            end = time.time()

            inference_time = end-start

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        video_out.release()
        print('\nExiting')

main()