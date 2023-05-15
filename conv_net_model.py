import numpy as np

import tensorflow as tf

INPUT_SHAPE = (64, 160, 3)

x = tf.placeholder('float', shape=[None , *INPUT_SHAPE])
y = tf.placeholder('float', shape=[None, 1])
keep_prob = tf.placeholder('float')

def conv_net(images, keep_prob):

    conv1Filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 24], mean=0, stddev=0.08))
    conv2Filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 24, 36], mean=0, stddev=0.08))
    conv3Filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 36, 64], mean=0, stddev=0.08))
    conv4Filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.08))
    conv5Filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], mean=0, stddev=0.08))

    images = images / 127.5 - 1.0

    #CONV LAYER 1
    conv1 = tf.nn.conv2d(images, conv1Filter,strides=[1,2,2,1],padding="SAME")
    #conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    conv1 = tf.layers.batch_normalization(conv1)
    conv1 = tf.nn.relu(conv1)

    #CONV LAYER 2
    conv2 = tf.nn.conv2d(conv1,conv2Filter,strides=[1,2,2,1],padding="SAME")
    #conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    conv2 = tf.layers.batch_normalization(conv2)
    conv2 = tf.nn.relu(conv2)

    #CONV LAYER 3
    conv3 = tf.nn.conv2d(conv2,conv3Filter,strides=[1,2,2,1],padding="SAME")
    #conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    conv3 = tf.layers.batch_normalization(conv3)
    conv3 = tf.nn.relu(conv3)

    #CONV LAYER 4
    conv4 = tf.nn.conv2d(conv3,conv4Filter,strides=[1,1,1,1],padding="SAME")
    #conv4 = tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    conv4 = tf.layers.batch_normalization(conv4)
    conv4 = tf.nn.relu(conv4)

    #CONV LAYER 5
    conv5 = tf.nn.conv2d(conv4,conv5Filter,strides=[1,1,1,1],padding="SAME")
    #conv5 = tf.nn.max_pool(conv5,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    conv5 = tf.layers.batch_normalization(conv5)
    conv5 = tf.nn.relu(conv5)

    #FLATTEN
    flat = tf.contrib.layers.flatten(conv5)


    #FULLY CONNECTED LAYER 1
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=100, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)

    #FULLY CONNECTED LAYER 2
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=50, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)

    #FULLY CONNECTED LAYER 3
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=10, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)

    #FULLY CONNECTED LAYER 4
    return tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1, activation_fn=None)
