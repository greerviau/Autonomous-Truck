import cv2
import time
import csv
import os
import sys
import pyvjoy
import pydirectinput

import numpy as np
import tensorflow as tf

from PIL import ImageGrab
from xbox_controller import XboxController
from conv_net_model import *
from tensorflow.keras.models import load_model
from utils import *
from get_keys import key_check


CONV_NET_MODEL = 'conv_net_models/conv_net_v29_medium_Peterbilt/conv_net.ckpt'

MAX_VJOY = 32767
RESOLUTION = (1280, 720)
MAX_SPEED = 50
TIME_TO_TELEPORT = 15
COLLECTING_DATA = False

xbox = XboxController()
joy = pyvjoy.VJoyDevice(1)

network = conv_net(x, keep_prob)
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
saver.restore(sess, CONV_NET_MODEL)

ocr = OCR()
teleporter = Teleporter()

brake_net = load_model('brake_net_model/brake_net.h5')

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

def get_frame():
    frame = grab_frame(region = (0,0,1280,720)) 
    frame = cv2.resize(frame, RESOLUTION)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def crop_road(frame):
    #GET THE ROAD SECTION THAT WILL BE INPUT TO THE CONV NET
    road = frame[380:630, 330:950, :]  #W, H = 620, 250
    #road = cv2.resize(road, (124, 50))
    #road = cv2.resize(road, (200, 80))
    road = cv2.resize(road, (160, 64))
    return road

def get_steering_prediction(frame):
    input_frame = np.copy(frame).astype(np.float16)

    #PREDICT THE TURN RADIUS
    pred_left_x = sess.run(network, feed_dict={x:[input_frame], keep_prob:1.0})[0][0]
    #TRANSLATE THE TURN RADIUS TO A JOYSTICK X COMMAND
    ai_joy_x = int(((1.0+pred_left_x)/2) * MAX_VJOY)
    return pred_left_x, ai_joy_x

def main():

    if not os.path.exists('test_videos'):
        os.makedirs('test_videos')
    else:
        videos = os.listdir('test_videos')
    
    vid_number = int(videos[len(videos)-1].split('_')[1].split('.')[0])

    #video_out = cv2.VideoWriter("test_videos/test_{}.mp4".format(vid_number+1),cv2.VideoWriter_fourcc(*'XVID'), 30, RESOLUTION)

    human_override = True
    lock = False
    lights_on = False
    changing_lanes = True
    timing = False

    light_avg = []

    inference_time = 1
    frame_count = 0
    speed = 0
    last_speed = 0
    lane_change = 0
    counter = 0

    timer = 0

    key_check()
    
    temp_queue = None
    
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
            w_key = 'W' in key_check()
            s_key = 'S' in key_check()

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
            if not lock:
                if human_override and (b_button or p_key):
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

                elif not human_override and (b_button or p_key or w_key or s_key):
                    human_override=True

                    if COLLECTING_DATA:
                        video_data.release()
                
                lock = True

            if (not b_button and not p_key and not w_key and not s_key) and lock:
                lock = False

            #GET THE GAME FRAME
            frame = get_frame()

            #UPDATE CURRENT SPEED EVERY 2 FRAMES
            current_speed = ocr.get_speed(frame)
            if current_speed is not None:
                speed = current_speed
                if not human_override:
                    if speed < 5 and not timing:
                        start_timer = time.time()
                        timing = True
                    elif speed < 5 and timing:
                        if time.time() - start_timer > TIME_TO_TELEPORT:
                            joy.data.wAxisYRot = MAX_VJOY//2
                            joy.update()
                            teleporter.teleport_to_next_location()
                            timing = False
                            #CENTER THE WHEELS
                            for i in range(5):
                                joy.data.wAxisX = 0
                                joy.update()
                                time.sleep(1)
                            for i in range(5):
                                joy.data.wAxisX = MAX_VJOY
                                joy.update()
                                time.sleep(1)
                            for i in range(5):
                                joy.data.wAxisX = 9500
                                joy.update()
                                time.sleep(1)
                    else:
                        timing = False
                else:
                    timing = False

            #UPDATE THE SPEED LIMIT EVERY N FRAMES
            if frame_count % 25 == 0:
                speed_limit = ocr.get_speed_limit(frame)
                frame_count = 0

            #SPEEDING
            speeding = speed - min(speed_limit, MAX_SPEED) > 2

            frame_count+=1

            #FRAME FOR COLLECTING DATA
            write_frame = np.copy(frame)
            
            #MEASURE THE LIGHT LEVEL FOR THE HEADLIGHTS
            light_avg.append(np.mean(frame[0,:,:]))
            if len(light_avg) > 50:
                light_avg.pop(0)

            light = np.mean(light_avg)

            road = crop_road(frame)
            '''
            gps = frame[511:611, 1030:1210, :]
            gps = cv2.resize(gps, (124, 50))

            road = np.concatenate((road, gps), axis=0)
            '''
            pred_left_x, ai_joy_x = get_steering_prediction(road)

            joy.data.wAxisY = MAX_VJOY // 2
            #joy.data.wAxisY = MAX_VJOY - int(((1.0+left_y)/2) * MAX_VJOY)

            #BRAKE PREDICTION
            
            brake_frame = cv2.cvtColor(road, cv2.COLOR_BGR2GRAY)
            brake_frame = cv2.resize(brake_frame, (124, 50))
            if temp_queue is None:
                temp_queue = Queue(brake_frame, 20)
            else:
                temp_queue.add(brake_frame)
            '''
            temporal = np.float32(cv2.merge((brake_frame, temp_queue.get(10), temp_queue.get(20))))
            brake_pred = brake_net.predict(np.array([temporal]))[0][0]
            brake = brake_pred > 0.7
            ai_brake = int(.75 * MAX_VJOY)
            '''
            brake_pred = 0
            brake = False

            def draw_ui_to_frame(frame, road, overlay=True):

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

                if overlay:
                    overlay = np.copy(frame)
                    cv2.rectangle(overlay, (0,350), (330,710), (0,0,0), -1)
                    frame = cv2.addWeighted(overlay, 0.6, frame, 1-0.6, 1)

                frame = cv2.resize(frame, (1920,1080))

                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame, 'Autopilot: ', (30,570), font, 1.2, (255,255,255), 2)
                cv2.putText(frame, auto_pilot_status, (225,570), font, 1.2, color, 2)

                cv2.putText(frame, 'Predicted Turn: {:.3f}'.format(pred_left_x), (30,615), font, 1.2, (255,255,255), 2)

                cv2.putText(frame, 'Lane Change: '+lane, (30,690), font, 1.2, (255,255,255), 2)

                cv2.putText(frame, 'Adaptive CC: ', (30,735), font, 1.2, (255,255,255), 2)
                cv2.putText(frame, acc_status, (285,735), font, 1.2, color, 2)

                cv2.putText(frame, 'Max Speed: {}'.format(MAX_SPEED), (30,780), font, 1.2, (255,255,255), 2)

                cv2.putText(frame, 'Speed Limit: {}'.format(speed_limit), (30,825), font, 1.2, (255,255,255), 2)

                cv2.putText(frame, 'Speed: {}'.format(speed), (30,870), font, 1.2, (255,255,255), 2)

                cv2.putText(frame, 'Pedal: ', (30,915), font, 1.2, (255,255,255), 2)
                if not brake and not speeding:
                    cv2.putText(frame, 'GAS', (165,915), font, 1.2, (0,255,0), 2)
                else:
                    cv2.putText(frame, 'BRAKE', (165,915), font, 1.2, (0,0,255), 2)

                cv2.putText(frame, 'Headlights: ', (30,990), font, 1.2, (255,255,255), 2)
                cv2.putText(frame, headlights, (255,990), font, 1.2, headlight_color, 2)

                cv2.putText(frame, 'Inf Time: {:.2f} fps'.format(1.0/inference_time), (30,1040), font, 1.2, (255,255,255), 2)

                if timing:
                    if TIME_TO_TELEPORT - (time.time()-start_timer) > 1:
                        text = 'Teleporting in: {:.0f} seconds'.format(TIME_TO_TELEPORT - (time.time()-start_timer))
                        (label_width, label_height), baseline = cv2.getTextSize(text, font, 1, 3)
                        cv2.putText(frame, text, (960-label_width//2,540), font, 1.5, (0,0,255), 3)
                    else:
                        text = 'Teleporting'
                        (label_width, label_height), baseline = cv2.getTextSize(text, font, 1, 3)
                        cv2.putText(frame, text, (960-label_width//2,540), font, 1.5, (0,255,0), 3)

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
                if light < 90 and not lights_on:
                    pydirectinput.press(['l', 'l'])
                    lights_on = True
                elif light > 90 and lights_on:
                    pydirectinput.press('l')
                    lights_on = False

                if brake:
                    joy.data.wAxisYRot = ai_brake
                elif speeding:
                    joy.data.wAxisYRot = int(MAX_VJOY * 0.85)
                elif (pred_left_x > 0.25 or pred_left_x < -0.3) and speed > min(speed_limit, MAX_SPEED)-5:
                    joy.data.wAxisYRot = int(MAX_VJOY * 0.75)
                elif (pred_left_x > 0.25 or pred_left_x < -0.3):
                    joy.data.wAxisYRot = int(MAX_VJOY * 0.5)
                elif speed < min(speed_limit, MAX_SPEED):
                    joy.data.wAxisYRot = 0
                else:
                    joy.data.wAxisYRot = int(MAX_VJOY * 0.5)
                

            else:
                joy.data.wAxisX = human_joy_x
                joy.data.wAxisYRot = MAX_VJOY//2

            # UPDATE THE VIRTUAL JOYSTICK
            joy.update()

            #SETUP THE VISUAL FRAME
            
            frame = draw_ui_to_frame(frame, road)
            cv2.imshow('Auto Truck',cv2.resize(frame, (1280, 720)))
            #video_out.write(frame)
            '''
            
            black_screen = np.zeros_like(frame)

            black_screen = draw_ui_to_frame(black_screen, road, overlay=False)
            cv2.imshow('Auto Truck Black',black_screen)
            #video_out.write(black_screen)
            '''

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

            end = time.time()

            inference_time = end-start

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        #video_out.release()
        print('\nExiting')

main()