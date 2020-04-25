import tensorflow as tf
import os
import cv2
import numpy as np
import random

#(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

#X_train = X_train.reshape(-1, 784)
#X_test = X_test.reshape(-1, 784)

X = []
Y = []

for label in os.listdir('digit_data'):
    images = os.listdir('digit_data/'+label)
    random.shuffle(images)
    for image in images[:100]:
        X.append(cv2.imread('digit_data/'+label+'/'+image, 0))
        Y.append(int(label))

X = np.array(X)
Y = np.array(Y)

np.random.seed(547)
np.random.shuffle(X)
np.random.seed(547)
np.random.shuffle(Y)

print(X.shape, Y.shape)

from sklearn.model_selection import train_test_split

X_train = X[:-100]
X_test = X[-100:]
Y_train = Y[:-100]
Y_test = Y[-100:]


print(X_train.shape, Y_train.shape)

'''
from sklearn.svm import SVC

svc = SVC(gamma=0.001, C=100)
svc.fit(X_train[:10000], Y_train[:10000])

y_pred = svc.predict(X_test)
'''

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import backend as K
import pickle

batch_size = 16
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(X_test[0].dtype)

# convert class vectors to binary class matrices
Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(X_test)

for y, y_hat in zip(Y_test, y_pred):
    print(np.argmax(y), np.argmax(y_hat))

model.save('digit_rec.h5')