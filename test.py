from __future__ import print_function
import cPickle as pickle
import tensorflow as tf
import numpy as np
import math


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



sess = tf.InteractiveSession()

with open('dataset.pickle','rb') as f:
    save = pickle.load(f)
    gaze = save['gaze']
    img = save['img']
    pose = save['pose']


tmp = np.zeros((len(gaze),2))
for i in range(len(gaze)):
    tmp[i,0]= math.asin(-gaze[i,1])
    tmp[i,1]=math.atan2(-gaze[i,0],-gaze[i,2])
gaze = tmp
train_gaze=gaze[0:200000,:]
train_img= img[0:200000,:,:]
train_pose=pose[0:200000,:]

valid_gaze=gaze[200001:210000,:]
valid_img=img[200001:210000,:,:]
valid_pose=pose[200001:210000,:]

test_gaze=gaze[210001:213658,:]
test_img=img[210001:213658,:,:]
test_pose=pose[210001:213658,:]
n_sample = len(train_img)

train_img=np.reshape(train_img,(len(train_img),2160))
valid_img=np.reshape(valid_img,(len(valid_img),2160))
test_img=np.reshape(test_img,(len(test_img),2160))

x = tf.placeholder(tf.float32, shape=[None, 2160])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5,5,1,20])
b_conv1 = bias_variable([20])
x_image = tf.reshape(x,[-1,36,60,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 20, 50])
b_conv2 = bias_variable([50])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([6 * 12 * 50, 500])
b_fc1 = bias_variable([500])

h_pool2_flat = tf.reshape(h_pool2, [-1, 6*12*50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

x_addtion = tf.placeholder(tf.float32,shape=[None,3])
h_fc1_drop=tf.concat(1,[h_fc1_drop,x_addtion])

W_fc2 = weight_variable([503, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cost = tf.reduce_mean(tf.pow(y_conv-y_,2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#sess.run(tf.global_variables_initializer())
tf.global_variables_initializer().run()
batch_size = 128
for i in range(20000):
    offset = (i * batch_size) % (train_gaze.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_img[offset:(offset + batch_size), :]
    add_batch_data = train_pose[offset:(offset + batch_size), :]
    batch_labels = train_gaze[offset:(offset + batch_size), :]

    #batch = mnist.train.next_batch(50)
    if i%10 == 0:
        train_cost = sess.run(cost,feed_dict={x: batch_data,x_addtion:add_batch_data, y_: batch_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_cost))
    train_step.run(feed_dict={x: batch_data,x_addtion:add_batch_data,  y_: batch_labels, keep_prob: 0.5})

print("test accuracy %g"%sess.run(cost,feed_dict={
    x: test_img,x_addtion:test_pose, y_: test_gaze, keep_prob: 1.0}))