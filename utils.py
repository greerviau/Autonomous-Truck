import cv2
import random
import time
import pydirectinput
import imutils

import numpy as np

from PIL import ImageGrab
from tensorflow.keras.models import load_model

class Teleporter(object):
    def __init__(self):
        self.LOCATIONS = {
            "Eureka Safe":"-116824;2.43652;-18526.5;-2.62944;-0.0014884",
            "San Diego North":"-99397.9;1.01643;24737;0.803268;0.0106427",
            "San Diego East":"-97625.1;10.7149;23799.8;-1.16027;-0.000403042",
            'Elko':'-79915.8;109.054;-20751.3;1.1608;-0.0114734',
            'Hornbrook':'-110169;64.0087;-30937.9;2.61471;-1.18249e-06',
            'San Simon':'-60921;62.6136;33147.5;1.27878;-0.000141385',
            'Las Vegas North':'-84200.4;41.469;6374.12;-0.21822;-0.000410509',
            'Las Vegas East':'-85262.4;31.2889;5454.31;0.772568;-0.000721538',
            'Holbrook':'-62099.4;90.7263;16640.4;1.73795;-0.00403668',
            'Los Angeles Coast':'-103991;20.9444;14413.6;1.00404;0.00877141'
        }

        self.GAS_STATION = "-110682;5.89365;-12413;-1.5789;-8.92041e-05"

        self.LOCATION_INDEX = 0

    def teleport_to_next_location(self):
        pydirectinput.press('e')
        time.sleep(1)
        self.teleport_to_location(self.GAS_STATION)   #Teleport to gas station
        time.sleep(2)
        pydirectinput.press('f9')   #Teleport truck
        time.sleep(5)
        pydirectinput.keyDown('enter')   #Fill gas tank for 20 seconds
        time.sleep(20)
        pydirectinput.keyUp('enter')

        time.sleep(5)

        loc = list(self.LOCATIONS)[self.LOCATION_INDEX]
        self.LOCATION_INDEX += 1
        if self.LOCATION_INDEX >= len(list(self.LOCATIONS)):
            self.LOCATION_INDEX = 0
            
        coordinates = self.LOCATIONS[loc]
        self.teleport_to_location(coordinates)   #Teleport to random location
        time.sleep(2)
        pydirectinput.press('f9')   #Teleport truck
        time.sleep(5)
        pydirectinput.press('4')    #Roof cam
        time.sleep(0.5)
        pydirectinput.press('e')
        time.sleep(1)

    def teleport_to_location(self, coordinates):

        pydirectinput.press('`')
        time.sleep(0.1)
        pydirectinput.press(['g','o','t','o','space'])

        pydirectinput.press([c for c in coordinates])

        time.sleep(0.1)
        pydirectinput.press('enter')
        time.sleep(2)
        pydirectinput.press('`')

    

class Queue():
    def __init__(self, frame, length):
        self.queue = [frame for i in range(length)]

    def add(self, frame):
        temp_frame = frame
        for i in range(len(self.queue)):
            temp_frame = self.queue[i]
            self.queue[i] = frame
            frame = temp_frame

    def get(self, index):
        return self.queue[index-1]

    def get_queue(self):
        return self.queue


class OCR():
    def __init__(self):
        self.ocr_model = load_model('digit_rec_model/digit_rec.h5')
        self.counter = 0
        

    def get_speed(self, frame):

        count, _ = self.count_digits(frame)
        if count == 1:
            speed_frame = frame[472:488,1003:1015]

            speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

            speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

            speed_frame_thresh = np.zeros_like(speed_frame)
            speed_frame_thresh[speed_frame > np.max(speed_frame)-70] = 255
            speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

            speed_frame_thresh = cv2.resize(speed_frame_thresh[5:23, 6:24], (28,28))

            #cv2.imshow('thresh_1', cv2.resize(speed_frame_thresh, (300,300)))

            return np.argmax(self.ocr_model.predict([[np.expand_dims(speed_frame_thresh.astype(np.float32)/ 255.0, axis=2)]])[0])

        elif count == 2:
            #SPD DIGIT 1
            speed_frame = frame[472:488,998:1009]

            speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

            speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

            speed_frame_thresh = np.zeros_like(speed_frame)
            speed_frame_thresh[speed_frame > np.max(speed_frame)-70] = 255

            speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

            speed_frame_thresh = cv2.resize(speed_frame_thresh[7:22, 12:28], (24,24))
            speed_1_final = np.zeros((28,28), dtype=np.uint8)

            speed_1_final[2:26,2:26] = speed_frame_thresh

            spd_1 = np.argmax(self.ocr_model.predict([[np.expand_dims(speed_1_final.astype(np.float32)/ 255.0, axis=2)]])[0])

            #cv2.imshow('thresh_1', cv2.resize(speed_1_final, (300,300)))

            #SPD DIGIT 2

            speed_frame = frame[472:488,1009:1016]

            speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

            speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

            speed_frame_thresh = np.zeros_like(speed_frame)
            speed_frame_thresh[speed_frame > np.max(speed_frame)-70] = 255

            speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

            speed_frame_thresh = cv2.resize(speed_frame_thresh[8:22, 0:24], (24,24))
            speed_2_final = np.zeros((28,28), dtype=np.uint8)

            speed_2_final[2:26,2:26] = speed_frame_thresh

            spd_2 = np.argmax(self.ocr_model.predict([[np.expand_dims(speed_2_final.astype(np.float32)/ 255.0, axis=2)]])[0])

            #cv2.imshow('thresh_2', cv2.resize(speed_2_final, (300,300)))
            return int('{}{}'.format(spd_1, spd_2))

        else:
            return None



    def get_speed_limit(self, frame):
        #SPD LIMIT 1

        speed_frame = frame[520:538,999:1008]

        speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

        #speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

        speed_frame_thresh = np.zeros_like(speed_frame)
        speed_frame_thresh[speed_frame < np.max(speed_frame)-80] = 255

        speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

        speed_frame_thresh = cv2.resize(speed_frame_thresh[6:22,4:26], (24,24))
        speed_limit_1_final = np.zeros((28,28), dtype=np.uint8)

        speed_limit_1_final[2:26,2:26] = speed_frame_thresh

        spd_lim_1 = np.argmax(self.ocr_model.predict([[np.expand_dims(speed_limit_1_final.astype(np.float32)/ 255.0, axis=2)]])[0])

        #cv2.imshow('thresh_3', cv2.resize(speed_limit_1_final, (300,300)))
        
        #SPD LIMIT 2

        speed_frame = frame[520:538,1008:1017]

        speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

        #speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

        speed_frame_thresh = np.zeros_like(speed_frame)
        speed_frame_thresh[speed_frame < np.max(speed_frame)-80] = 255

        speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

        speed_frame_thresh = cv2.resize(speed_frame_thresh[6:22,:20], (24,24))
        speed_limit_2_final = np.zeros((28,28), dtype=np.uint8)

        speed_limit_2_final[2:26,2:26] = speed_frame_thresh

        spd_lim_2 = np.argmax(self.ocr_model.predict([[np.expand_dims(speed_limit_2_final.astype(np.float32)/ 255.0, axis=2)]])[0])

        #cv2.imshow('thresh_4', cv2.resize(speed_limit_2_final, (300,300)))
        
        #print(spd_1, spd_2)

        speed_limit = int('{}{}'.format(spd_lim_1, spd_lim_2))

        return speed_limit

    def count_digits(self, frame):
        digit = frame[474:486,1006:1012].copy()
        digits_frame = cv2.resize(cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY), (144,128))
        thresh = np.zeros_like(digits_frame)
        thresh[digits_frame > 170] = 255
        #cv2.imshow('digit', cv2.resize(digit,(300,300)))
        #cv2.imshow('count', cv2.resize(thresh,(300,300)))

        digit_count = 1

        if np.sum(thresh[:,0]) > 0 and np.sum(thresh[:,-1]) > 0:
            digit_count = 2

        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        digitCnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            if w >= 10 and (h >= 50 and h <= 120):
                cv2.rectangle(digits_frame, (x,y), (x+w, y+h), (0,255,0), 1)
                digitCnts.append(c)

        digit_count = len(digitCnts)
        '''
        return digit_count, digits_frame

def grab_frame(region = (0,30,1280,740)): 
    frame =  np.array(ImageGrab.grab(bbox=region))
    return frame

def get_curve(rng, x_d, y_d):
    point_list = []
    j = 1/100
    for i in range(100):
        t = float(j*i)*(np.pi/2)
        #print(t)
        x = np.cos(t)*(rng*150)
        
        y = np.sin(t)*300
        X = x+x_d
        if rng < 0:
            X+=abs(rng*150)
        else:
            X-=abs(rng*150)
        Y = y_d-y
        point_list.append((X, Y))
    #print(point_list)
    pts = np.array(point_list).astype(np.int32)
    return pts

if __name__ == "__main__":
    #teleport_to_random_location()
    cap = cv2.VideoCapture('data/roof_cam/raw/sess_50/clip_4/drive.mp4')

    mini = 2
    maxi = 2

    ocr = OCR()
    count = 10000
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        speed = ocr.get_speed_limit(frame)
        print(speed)

        cv2.imshow('frame', frame)
        count+=1
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    