import cv2
import time
import csv
import os
import sys
import pyvjoy

import numpy as np
import tensorflow as tf

from PIL import ImageGrab
from xbox_controller import XboxController
from utils import grab_frame
from conv_net_model import *
from tensorflow.keras.models import load_model
from utils import get_speed, get_speed_limit, get_curve
from direct_keys import PressKey, ReleaseKey, W, A, S, D, L
from get_keys import key_check


CONV_NET_MODEL = 'conv_net_models/conv_net_v24_Peterbilt/conv_net.ckpt'

MAX_VJOY = 32767
RESOLUTION = (1280, 720)
MAX_SPEED = 50
COLLECTING_DATA = False

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


def main():

    if not os.path.exists('test_videos'):
        os.makedirs('test_videos')
    else:
        videos = os.listdir('test_videos')
    
    vid_number = int(videos[len(videos)-1].split('_')[1].split('.')[0])

    video_out = cv2.VideoWriter("test_videos/test_{}.mp4".format(vid_number+1),cv2.VideoWriter_fourcc(*'XVID'), 30, RESOLUTION)

    xbox = XboxController()
    joy = pyvjoy.VJoyDevice(1)

    network = conv_net(x, keep_prob)
    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    saver.restore(sess, CONV_NET_MODEL)

    ocr_model = load_model('digit_rec_model/digit_rec.h5')

    #brake_net = load_model('brake_net_model/brake_net.h5')

    human_override = True
    lock = False
    lights_on = False
    changing_lanes = True

    light_avg = []

    inference_time = 1
    frame_count = 0
    speed_limit = 0
    lane_change = 0
    counter = 0

    key_check()
    
    if COLLECTING_DATA:
        clip = 1

        data_dir = os.path.join('data/roof_cam/raw_autonomous', sys.argv[1], 'clip_'+str(clip))

        if os.path.exists(data_dir):
            print(data_dir, ' Already Exists!')
            return
        else:
            os.makedirs(data_dir)

        video_data = cv2.VideoWriter(os.path.join(data_dir, 'drive.mp4'),cv2.VideoWriter_fourcc(*'XVID'), 30, RESOLUTION)

        outputfile = open(os.path.join(data_dir, 'controls.csv'), 'w', newline='')
        csvwriter = csv.writer(outputfile)
        csvwriter.writerow(['Left_X', 'LT'])

    try:

        while True:

            start = time.time()

            #GET THE INPUTS FROM THE XBOX GAMEPAD
            input_dict = xbox.read()

            #RESET THE VIRTUAL JOYSTICK
            joy.reset()

            set_joy_from_gamepad(joy, input_dict)

            #MEASURE THE TURN RADIUS FROM THE GAMEPAD
            left_x = input_dict['Left_X']

            #TRANSLATE THE TURN RADIUS TO A JOYSTICK COMMAND
            human_joy_x = int(((1.0+left_x)/2) * MAX_VJOY)

            #left_y = input_dict['Left_Y']

            b_button = input_dict['B']
            left_bump = input_dict['LB']
            right_bump = input_dict['RB']

            p_key = 'P' in key_check()

            #CHANGE LANES
            if left_bump and not changing_lanes:
                lane_change = -1
                changing_lanes = True
            elif right_bump and not changing_lanes:
                lane_change = 1
                changing_lanes = True

            if changing_lanes:
                if counter < 30:
                    counter += 1
                else:
                    counter = 0
                    changing_lanes = False
                    lane_change = 0

            #ENGAGE OR DISENGAGE THE AUTOPILOT
            if (b_button or p_key) and not lock:
                if human_override:
                    human_override=False

                    #MAKE A NEW DATA COLLECTION CLIP
                    if COLLECTING_DATA:
                        data_dir = os.path.join('data/roof_cam/raw_autonomous', sys.argv[1], 'clip_'+str(clip))
                        if not os.path.exists(data_dir):
                            os.makedirs(data_dir)

                        video_data = cv2.VideoWriter(os.path.join(data_dir, 'drive.mp4'),cv2.VideoWriter_fourcc(*'XVID'), 30, RESOLUTION)

                        outputfile = open(os.path.join(data_dir, 'controls.csv'), 'w', newline='')
                        csvwriter = csv.writer(outputfile)
                        csvwriter.writerow(['Left_X', 'LT'])

                        clip += 1

                elif not human_override:
                    ReleaseKey(W)
                    ReleaseKey(S)
                    human_override=True

                    if COLLECTING_DATA:
                        video_data.release()
                
                lock = True

            if (not b_button and not p_key) and lock:
                lock = False

            #GET THE GAME FRAME
            frame = grab_frame(region = (0,0,1280,720)) 
            frame = cv2.resize(frame, RESOLUTION)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            write_frame = np.copy(frame)
            
            #MEASURE THE LIGHT LEVEL FOR THE HEADLIGHTS
            light_avg.append(np.mean(frame[0,:,:]))
            if len(light_avg) > 50:
                light_avg.pop(0)

            light = np.mean(light_avg)

            #GET THE ROAD SECTION THAT WILL BE INPUT TO THE CONV NET
            road = frame[380:630, 330:950, :]  #W, H = 620, 250
            combined = cv2.resize(road, (124, 50))
            '''
            gps = frame[511:611, 1030:1210, :]
            gps = cv2.resize(gps, (124, 50))

            combined = np.concatenate((road, gps), axis=0)
            '''
            input_frame = np.copy(combined).astype(np.float16)

            #PREDICT THE TURN RADIUS
            pred_left_x = sess.run(network, feed_dict={x:[input_frame], keep_prob:1.0})[0][0]
            #TRANSLATE THE TURN RADIUS TO A JOYSTICK X COMMAND
            ai_joy_x = int(((1.0+pred_left_x)/2) * MAX_VJOY)

            joy.data.wAxisY = MAX_VJOY // 2
            #joy.data.wAxisY = MAX_VJOY - int(((1.0+left_y)/2) * MAX_VJOY)

            # UPDATE THE VIRTUAL JOYSTICK
            joy.update()

            #BRAKE PREDICTION
            '''
            brake_pred = brake_net.predict(np.array([input_frame]))[0][0]
            brake = brake_pred > 0.85
            '''
            brake_pred = 0
            brake = False
            
            #UPDATE CURRENT SPEED EVERY 2 FRAMES
            if frame_count % 2 == 0:
                speed = get_speed(frame, ocr_model)

            #UPDATE THE SPEED LIMIT EVERY 50 FRAMES
            if frame_count % 50 == 0:
                speed_limit = get_speed_limit(frame, ocr_model)
                frame_count = 0

            frame_count+=1

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

            lane = 'None'
            if lane_change == -1:
                lane = 'Left'
            elif lane_change == 1:
                lane = 'Right'

            def draw_text_to_frame(frame):

                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame, 'Autopilot: ', (20,350), font, 0.8, (255,255,255), 2)
                cv2.putText(frame, auto_pilot_status, (150,350), font, 0.8, color, 2)

                cv2.putText(frame, 'Predicted Turn: {:.3f}'.format(pred_left_x), (20,380), font, 0.8, (255,255,255), 2)

                cv2.putText(frame, 'Joy X: '+str(joy.data.wAxisX), (20,410), font, 0.8, (255,255,255), 2)

                cv2.putText(frame, 'Lane Change: '+lane, (20,440), font, 0.8, (255,255,255), 2)

                cv2.putText(frame, 'Auto CC: ', (20,490), font, 0.8, (255,255,255), 2)
                cv2.putText(frame, acc_status, (140,490), font, 0.8, color, 2)

                cv2.putText(frame, 'Max Speed: {}'.format(MAX_SPEED), (20,520), font, 0.8, (255,255,255), 2)

                cv2.putText(frame, 'Speed Limit: {}'.format(speed_limit), (20,550), font, 0.8, (255,255,255), 2)

                cv2.putText(frame, 'Speed: {}'.format(speed), (20,580), font, 0.8, (255,255,255), 2)

                if not brake:
                    cv2.putText(frame, 'GAS ', (20,630), font, 0.8, (0,255,0), 2)
                else:
                    cv2.putText(frame, 'BRAKE ', (20,630), font, 0.8, (0,0,255), 2)

                cv2.putText(frame, 'Headlights: ', (20,660), font, 0.8, (255,255,255), 2)
                cv2.putText(frame, headlights, (170,660), font, 0.8, headlight_color, 2)

                cv2.putText(frame, 'Inf Time: {:.2f} fps'.format(1.0/inference_time), (20,690), font, 0.8, (255,255,255), 2)

                return frame

            
            if not human_override:

                joy.data.wAxisX = ai_joy_x

                if lane_change == -1:
                    joy.data.wAxisX = ai_joy_x - int(0.1 * MAX_VJOY)
                elif lane_change == 1:
                    joy.data.wAxisX = ai_joy_x + int(0.1 * MAX_VJOY)

                #COLLECT DATA FROM THE DRIVE
                if COLLECTING_DATA:
                    video_data.write(write_frame)
                    csvwriter.writerow([pred_left_x, brake_pred])

                #TURN THE HEADLIGHTS ON AND OFF DEPENDING ON LIGHT LEVEL
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

                if brake or speed - min(speed_limit, MAX_SPEED) > 5:
                    ReleaseKey(W)
                    PressKey(S)
                    brake = True
                elif speed < min(speed_limit, MAX_SPEED):
                    ReleaseKey(S)
                    PressKey(W)
                else:
                    ReleaseKey(S)
                    ReleaseKey(W)
                

            else:
                joy.data.wAxisX = human_joy_x

            #SETUP THE VISUAL FRAME
            shp = combined.shape
            combined = cv2.resize(combined, (shp[1]*2, shp[0]*2))
            shp = combined.shape
            
            overlay = np.copy(frame)
            cv2.rectangle(overlay, (0,320), (330,710), (0,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 1-0.6, 1)
            frame = draw_text_to_frame(frame)

            frame[0:shp[0], 640-shp[1]//2:640+shp[1]//2, :] = combined
            cv2.imshow('Auto Truck',frame)
            video_out.write(frame)
            
            ''' 
            black_screen = np.zeros_like(frame)
            black_screen[0:shp[0], 640-shp[1]//2:640+shp[1]//2, :] = road

            black_screen = draw_text_to_frame(black_screen)
            cv2.imshow('Auto Truck Black',black_screen)
            video_out.write(black_screen)
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