from PIL import ImageGrab
import cv2
import numpy as np

def grab_frame(region = (0,30,1280,740)): 
    frame =  np.array(ImageGrab.grab(bbox=region))
    return frame

def get_speed(frame, ocr_model):
    #SPD DIGIT 1
    speed_frame = frame[477:493,1002:1012]

    speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

    speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

    speed_frame_thresh = np.zeros_like(speed_frame)
    speed_frame_thresh[speed_frame > np.max(speed_frame)-70] = 255

    speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

    speed_frame_thresh = cv2.resize(speed_frame_thresh[7:22, 12:28], (24,24))
    speed_1_final = np.zeros((28,28), dtype=np.uint8)

    speed_1_final[2:26,:24] = speed_frame_thresh

    spd_1 = np.argmax(ocr_model.predict([[np.expand_dims(speed_1_final.astype(np.float32)/ 255.0, axis=2)]])[0])

    #cv2.imshow('thresh_1', cv2.resize(speed_1_final, (300,300)))

    #SPD DIGIT 2

    speed_frame = frame[477:493,1012:1020]

    speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

    speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

    speed_frame_thresh = np.zeros_like(speed_frame)
    speed_frame_thresh[speed_frame > np.max(speed_frame)-70] = 255

    speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

    speed_frame_thresh = cv2.resize(speed_frame_thresh[8:22, 0:24], (24,24))
    speed_2_final = np.zeros((28,28), dtype=np.uint8)

    speed_2_final[2:26,2:26] = speed_frame_thresh

    spd_2 = np.argmax(ocr_model.predict([[np.expand_dims(speed_2_final.astype(np.float32)/ 255.0, axis=2)]])[0])

    #cv2.imshow('thresh_2', cv2.resize(speed_2_final, (300,300)))

    speed = int('{}{}'.format(spd_1, spd_2))

    return speed


def get_speed_limit(frame, ocr_model):
    #SPD LIMIT 1

    speed_frame = frame[590:608,1000:1009]

    speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

    speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

    speed_frame_thresh = np.zeros_like(speed_frame)
    speed_frame_thresh[speed_frame < 170] = 255

    speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

    speed_frame_thresh = cv2.resize(speed_frame_thresh[6:22,4:26], (24,24))
    speed_limit_1_final = np.zeros((28,28), dtype=np.uint8)

    speed_limit_1_final[2:26,2:26] = speed_frame_thresh

    spd_lim_1 = np.argmax(ocr_model.predict([[np.expand_dims(speed_limit_1_final.astype(np.float32)/ 255.0, axis=2)]])[0])

    #cv2.imshow('thresh_3', cv2.resize(speed_limit_1_final, (300,300)))

    #SPD LIMIT 2

    speed_frame = frame[590:608,1009:1018]

    speed_frame = cv2.cvtColor(speed_frame, cv2.COLOR_BGR2GRAY)

    speed_frame = cv2.resize(speed_frame, (speed_frame.shape[1]*5, speed_frame.shape[0]*5))

    speed_frame_thresh = np.zeros_like(speed_frame)
    speed_frame_thresh[speed_frame < 170] = 255

    speed_frame_thresh = cv2.resize(speed_frame_thresh, (28,28))

    speed_frame_thresh = cv2.resize(speed_frame_thresh[6:22,:20], (24,24))
    speed_limit_2_final = np.zeros((28,28), dtype=np.uint8)

    speed_limit_2_final[2:26,2:26] = speed_frame_thresh

    spd_lim_2 = np.argmax(ocr_model.predict([[np.expand_dims(speed_limit_2_final.astype(np.float32)/ 255.0, axis=2)]])[0])

    #cv2.imshow('thresh_4', cv2.resize(speed_limit_2_final, (300,300)))

    #print(spd_1, spd_2)

    speed_limit = int('{}{}'.format(spd_lim_1, spd_lim_2))

    return speed_limit



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