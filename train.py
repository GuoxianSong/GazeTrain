from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression

import matplotlib
import scipy.io as sio
import cPickle as pickle
import tensorflow as tf
import numpy as np
import math
#
# train_gaze = []
# train_img = []
# train_pose = []
#
# valid_gaze = []
# valid_img = []
# valid_pose =[]
#
# test_gaze = []
# test_img = []
# test_pose = []


with open('dataset.pickle','rb') as f:
    save = pickle.load(f)
    gaze = save['gaze']
    img = save['img']
    pose = save['pose']
        #gaze, img, pose=pickle.load(f)


tmp = np.zeros((len(gaze),2))
for i in range(len(gaze)):
    tmp[i,0]= math.asin(-gaze[i,1])
    tmp[i,1]=math.atan2(-gaze[i,0],-gaze[i,2])
gaze = tmp
train_gaze=gaze[0:200000,:]
#train_gaze=np.reshape(train_gaze,(200000,1))
train_img= img[0:200000,:,:]
train_pose=pose[0:200000,:]

valid_gaze=gaze[200001:210000,0]
valid_img=img[200001:210000,:,:]
valid_pose=pose[200001:210000,:]

test_gaze=gaze[210001:213658,0]
test_img=img[210001:213658,:,:]
test_pose=pose[210001:213658,:]
n_sample = len(train_img)

train_img=np.reshape(train_img,(len(train_img),2160))
valid_img=np.reshape(valid_img,(len(valid_img),2160))
test_img=np.reshape(test_img,(len(test_img),2160))

x = tf.placeholder(tf.float32,[None,2160]);
W = tf.Variable(tf.zeros([2160,1]));
#x_addition = tf.placeholder(tf.float32,[None,3])
#W_addition = tf.Variable(tf.zeros([3,1]))


b = tf.Variable(tf.zeros([1]));

y =tf.matmul(x,W)+b #tf.matmul(x_addition,W_addition);
y_= tf.placeholder(tf.float32,[None,1]);


loss = tf.reduce_sum(tf.pow(y-y_,2))/2*n_sample

train_step= tf.train.GradientDescentOptimizer(0.01).minimize(loss)


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1000):
        if i%10 ==0:
            c=sess.run(loss,feed_dict={x: train_img, y_: train_gaze})
            print("cost=", "{:.9f}".format(c))
        sess.run(train_step,feed_dict={x:train_img,y_:train_gaze})


   # print(sess.run(loss, feed_dict={x: test_img,x_addition:test_pose, y_: test_gaze}))
