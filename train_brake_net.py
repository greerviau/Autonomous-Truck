#import tensorflow as tf
import os
import cv2
import numpy as np
import random
import pandas as pd

processed_data_dir = 'data/roof_cam/processed_braking/Peterbilt'

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

data_dir = 'data/roof_cam/raw/Peterbilt'

session_folders = os.listdir(data_dir)
'''
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

        wheel_data = pd.read_csv(os.path.join(split_dir, split, 'controls.csv'))['LT'].to_numpy()

        if sum(wheel_data) > 0:
            wheel_data[wheel_data > 0] = 1

            control_data.append(wheel_data)

            print('Size: ',len(wheel_data))

            while True:
                ret, frame = road_video.read()

                if not ret:
                    break

                road = frame[380:630, 330:950, :]  #W, H = 620, 250
                road = cv2.resize(road, (124, 50))
            
                #normal_frame = frame / 127.5 - 1.0
                
                total_frames.append(np.copy(road).astype(np.float16))

                cv2.imshow('Frame', road)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            road_video.release()
            cv2.destroyAllWindows()


    control_data = np.concatenate(control_data, axis=0)
    total_frames = np.array(total_frames)

    #total_frames, control_data = ballance(total_frames, control_data)

    control_data = pd.DataFrame(control_data, columns=['Brake'])
    np.save(os.path.join(session_data_dir, 'X'), total_frames)
    control_data.to_csv(os.path.join(session_data_dir, 'Y.csv'))
'''

aggregate_frames = []
aggregate_data = []

session_folders = os.listdir(processed_data_dir)

print('Reloading...')
for session in session_folders:
    session_dir = os.path.join(processed_data_dir, session)
    aggregate_frames.extend(list(np.load(os.path.join(session_dir, 'X.npy'))))
    aggregate_data.extend(list(pd.read_csv(os.path.join(session_dir, 'Y.csv'))['Brake'].to_numpy()))

brake = []
not_brake = []

for i in range(len(aggregate_data)):
    if aggregate_data[i] == 0:
        not_brake.append([np.copy(aggregate_frames[i]), aggregate_data[i]])
    else:
        brake.append([np.copy(aggregate_frames[i]), aggregate_data[i]])

aggregate_frames = None
aggregate_data = None

print('Before Ballance')
print('Brake: ',len(brake))
print('No Brake: ',len(not_brake))

random.shuffle(brake)
random.shuffle(not_brake)

limit = len(brake)

brake = np.array(brake)
not_brake = np.array(not_brake[:limit])

print('After Ballance')
print('Brake: ',len(brake))
print('No Brake: ',len(not_brake))

combined = np.concatenate((brake, not_brake), axis=0)

X = list(combined[:,0])
Y = list(combined[:,1])


Y = pd.DataFrame(Y, columns=['Brake'])
print('Num frames: {} Data length: {}'.format(len(X), len(Y)))
X = np.array(X)
print('Frames Shape: ',X.shape)

np.save(os.path.join(processed_data_dir, 'X'), X)
Y.to_csv(os.path.join(processed_data_dir, 'Y.csv'))


X = np.load('data/roof_cam/processed_braking/Peterbilt/X.npy')
Y = pd.read_csv('data/roof_cam/processed_braking/Peterbilt/Y.csv')['Brake'].to_numpy()

#Y = Y.reshape(-1,1)

print(X.shape, Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
import pickle

model = Sequential()
model.add(Lambda(lambda x: x/255.0))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, epochs=50, verbose=1, validation_data=(X_test, Y_test))

model.save('brake_net_model/brake_net.h5')
