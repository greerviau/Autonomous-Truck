import numpy as np 
import pandas as pd 
import sys
import csv
import os
import cv2
import pickle
import scipy
import random

def ballance(frames, data):
    print('Before Ballancing')
    print('Num frames: {} Data length: {}'.format(len(frames), len(data)))

    print('Ballancing')

    left = []
    right = []
    left_straight = []
    right_straight = []
    straight = []

    for i in range(len(frames)):
        if data[i] < -0.15:
            left.append([np.copy(frames[i]), data[i]])
        elif data[i] > 0.15:
            right.append([np.copy(frames[i]), data[i]])
        elif data[i] < 0:
            left_straight.append([np.copy(frames[i]), data[i]])
        elif data[i] > 0:
            right_straight.append([np.copy(frames[i]), data[i]])
        else:
            straight.append([np.copy(frames[i]), data[i]])

    frames = None
    data = None

    print('Left Samples: ',len(left))
    print('Right Samples: ',len(right))
    print('Left Straight Samples: ',len(left_straight))
    print('Right Straight Samples: ',len(left_straight))
    print('Straight Samples: ',len(straight))

    print('Shuffling')

    random.shuffle(left)
    random.shuffle(right)
    random.shuffle(left_straight)
    random.shuffle(right_straight)
    random.shuffle(straight)

    limit = min(len(left), len(right), len(left_straight), len(right_straight), len(straight))

    left = left[:limit]
    right = right[:limit]
    left_straight = left_straight[:limit]
    right_straight = right_straight[:limit]
    straight = straight[:limit]
    print('Left Samples: ',len(left))
    print('Right Samples: ',len(right))
    print('Left Straight Samples: ',len(left_straight))
    print('Right Straight Samples: ',len(left_straight))
    print('Straight Samples: ',len(straight))

    left = np.array(left)
    right = np.array(right)
    left_straight = np.array(left_straight)
    right_straight = np.array(right_straight)
    straight = np.array(straight)

    combined = np.concatenate((left, right, left_straight, right_straight, straight), axis=0)

    frames = list(combined[:,0])
    data = list(combined[:,1])

    frames = np.array(frames)
    data = np.array(data)

    return frames, data


processed_data_dir = 'data/roof_cam/processed_road'

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

data_dir = 'data/roof_cam/raw/Peterbilt'

session_folders = os.listdir(data_dir)

for session in session_folders:
    print()
    print(session)
    split_dir = os.path.join(data_dir, session)
    splits = os.listdir(split_dir)

    #print(aggregate_data.head())

    session_data_dir = os.path.join(processed_data_dir, session)

    if not os.path.exists(session_data_dir):
        os.makedirs(session_data_dir)

    total_frames = []
    control_data = []

    for split in splits:
        print(split)
        road_video = cv2.VideoCapture(os.path.join(split_dir, split, 'drive.mp4'))

        wheel_data = pd.read_csv(os.path.join(split_dir, split, 'controls.csv'))['Left_X'].to_numpy()

        wheel_data[(wheel_data < 0.05) & (wheel_data > -0.05)] = 0.0

        control_data.append(wheel_data)

        print('Size: ',len(wheel_data))

        while True:
            ret, frame = road_video.read()

            if not ret:
                break

            road = frame[380:630, 330:950, :]  #W, H = 620, 250
            road = cv2.resize(road, (124, 50))

            #gps = frame[511:611, 1030:1210, :]
            #gps = cv2.resize(gps, (124, 50))
        
            #normal_frame = frame / 127.5 - 1.0

            #combined = np.concatenate((road, gps), axis=0)
            
            total_frames.append(road.astype(np.float16))

            '''
            cv2.imshow('Visual', road)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''

        road_video.release()
        cv2.destroyAllWindows()


    control_data = np.concatenate(control_data, axis=0)
    total_frames = np.array(total_frames)

    total_frames, control_data = ballance(total_frames, control_data)

    control_data = pd.DataFrame(control_data, columns=['Left X'])
    np.save(os.path.join(session_data_dir, 'X'), total_frames)
    control_data.to_csv(os.path.join(session_data_dir, 'Y.csv'))


print('Reloading')
session_folders = os.listdir(processed_data_dir)
#session_folders = ['sess_06','sess_07','sess_08','sess_09', 'sess_10', 'sess_11']

aggregate_frames = []
aggregate_data = []


for session in session_folders:
    session_dir = os.path.join(processed_data_dir, session)
    aggregate_frames.extend(list(np.load(os.path.join(session_dir, 'X.npy'))))
    aggregate_data.extend(list(pd.read_csv(os.path.join(session_dir, 'Y.csv'))['Left X'].to_numpy()))

#aggregate_frames, aggregate_data = ballance(aggregate_frames, aggregate_data)


aggregate_data = pd.DataFrame(aggregate_data, columns=['Left X'])
print('Num frames: {} Data length: {}'.format(len(aggregate_frames), len(aggregate_data)))
aggregate_frames = np.array(aggregate_frames)
print('Frames Shape: ',aggregate_frames.shape)


np.save(os.path.join(processed_data_dir, 'X'), aggregate_frames)
aggregate_data.to_csv(os.path.join(processed_data_dir, 'Y.csv'))

