import pandas as pd
import numpy as np
import cv2
import os
import random as rnd

import matplotlib.pyplot as plt

from conv_net_model import *
from sklearn.model_selection import train_test_split


def train(X_train, Y_train, X_val, Y_val, save_path=None, epochs=20, batch_size=8, opt=None, lss=None, save=None, augment=False, retrain=False):
    batch_num = len(X_train)//batch_size
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        if retrain:
            save.restore(sess, save_path+'/conv_net.ckpt')
        else:
            sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            avg_loss = []
            samples_shifted = 0
            avg_shift = []
            for batch in range(batch_num):
                b_start = batch*batch_size
                b_end = (batch+1)*batch_size
                batch_x = np.copy(X_train[b_start:b_end])
                batch_y = Y_train[b_start:b_end].reshape(-1,1)

                if augment:
                    for i in range(len(batch_x)):
                        if rnd.random() < 0.1:
                            samples_shifted += 1

                            hsv = cv2.cvtColor(batch_x[i].astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.int16)
                            value = int(((rnd.random()*2 - 1)*255)*0.2)

                            avg_shift.append(value)

                            h, s, v = cv2.split(hsv)
                            v += value
                            v[v > 255] = 255
                            v[v < 0] = 0
                            hsv = cv2.merge((h, s, v))
                            batch_x[i] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float16)

                _, train_loss = sess.run([opt, lss], feed_dict={x:batch_x, y:batch_y, keep_prob:0.8})
                avg_loss.append(train_loss)
                print('\rEpoch {}/{} - Batch {}/{} - Loss: {:.5f} - Avg Loss: {:.5f} '.format(epoch+1, epochs, batch+1, batch_num, train_loss, np.mean(avg_loss)),end='')
                if augment:
                    print('- Samples Augmented: {} - Avg Shift: {:.2f} '.format(samples_shifted, np.mean(avg_shift)), end='')
            
            avg_val_loss = []
            for batch in range(len(X_val)//batch_size):
                b_start = batch*batch_size
                b_end = (batch+1)*batch_size
                val_loss = sess.run(lss, feed_dict={x:X_train[b_start:b_end], y:Y_train[b_start:b_end].reshape(-1,1), keep_prob:1.0})
                avg_val_loss.append(val_loss)

            print('- Val Loss: {:.5f} '.format(np.mean(avg_val_loss)), end='')
            
            print()

            np.random.seed(547)
            np.random.shuffle(X_train)
            np.random.seed(547)
            np.random.shuffle(Y_train)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save.save(sess, save_path+'/conv_net.ckpt')
    

def test(X_test, Y_test, network, saver, save_path=None):
    #Testing

    print(X_test.shape, Y_test.shape)

    X = X_test[:1000]
    Y = Y_test[:1000]

    np.random.seed(547)
    np.random.shuffle(X)
    np.random.seed(547)
    np.random.shuffle(Y)

    y_pred = []
    with tf.Session() as sess:
        saver.restore(sess, save_path+'/conv_net.ckpt')
        for frame in X:
            pred = sess.run(network, feed_dict={x:[frame], keep_prob:1.0})[0][0]
            y_pred.append(pred)

    plt.plot([i for i in range(300)], Y[:300])
    plt.plot([i for i in range(300)], y_pred[:300])
    plt.show()


def main():

    SAVE_PATH = 'conv_net_models/conv_net_v25_Peterbilt'
    MAX_EPOCHS = 1000
    BATCH_SIZE = 16
    AUGMENT = False
    
    network = conv_net(x, keep_prob)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(network, y))))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)
    saver = tf.train.Saver()

    print('Loading Data...')
    
    X = np.load('data/roof_cam/processed_road/X.npy')
    Y = pd.read_csv('data/roof_cam/processed_road/Y.csv')['Left X'].to_numpy()
    '''
    X = []
    Y = []

    data_dir = 'data/roof_cam/processed'

    for session in os.listdir(data_dir):
        X.append(np.load(os.path.join(data_dir, session, 'X.npy')))
        Y.append(pd.read_csv(os.path.join(data_dir, session, 'Y.csv'))['Left X'].to_numpy())
    
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    '''
    print(X.shape, Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    train(X_train, Y_train, X_test, Y_test, save_path=SAVE_PATH, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, opt=optimizer, lss=loss, save=saver, augment=AUGMENT, retrain=False)
    
    test(X, Y, network, saver, save_path=SAVE_PATH)

main()