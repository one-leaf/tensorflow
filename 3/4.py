# coding=utf-8
'''
增加了图表输出的功能，运行后，会在当前目录创建 train 的文件夹，存放了图表输出数据
执行 tensorboard --logdir=train/ 后，去浏览器访问 http://127.0.0.1:6006 即可。
'''
import tensorflow as tf
import numpy as np
import os
import random

x_one_len = 20
x_len=x_one_len*5

def batch(batch_size):
    train_x = []
    train_y = []
    for i in range(batch_size):
        x1 = random.randint(0, 500)
        x2 = random.randint(0, 500)
        y = [0, 1]
        if ((x1 < 200 and x2 < 200) or (x1 > 300 and x2 > 300)) and (x1<x2*2 and x1>x2*.05):
            y = [1, 0]
        x=np.binary_repr(x1, width=x_one_len)+np.binary_repr(x2, width=x_one_len)\
          +np.binary_repr(x1*x2, width=x_one_len)+np.binary_repr(x1*x1, width=x_one_len)\
          +np.binary_repr(x2*x2, width=x_one_len)
        x_=list(map(int, x))
        train_x.append(x_)
        train_y.append(y)
    return np.array(train_x), np.array(train_y)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 1, 1],
                        strides=[1, 1, 1, 1], padding='SAME')
with tf.name_scope('input'):
    x = tf.placeholder("float", shape=[None, x_len])
    y_ = tf.placeholder("float", shape=[None, 2])

sess = tf.InteractiveSession()

with tf.name_scope('conv-pool-1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1,10,10,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    tf.summary.histogram('W_conv1',W_conv1)
    tf.summary.histogram('b_conv1',b_conv1)

with tf.name_scope('conv-pool-2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    tf.summary.histogram('W_conv2',W_conv2)
    tf.summary.histogram('b_conv2',b_conv2)

with tf.name_scope('full-connect-1'):
    W_fc1 = weight_variable([10 * 10 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    tf.summary.histogram('W_fc1',W_fc1)
    tf.summary.histogram('b_fc1',b_fc1)

with tf.name_scope('full-connect-2'):
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    tf.summary.histogram('W_fc2',W_fc2)
    tf.summary.histogram('b_fc2',b_fc2)

with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    tf.summary.scalar('loss', cross_entropy)

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

with tf.name_scope('acc'):   
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar('acc', accuracy)

merged = tf.summary.merge_all()
current_dir = os.path.dirname(__file__)
train_writer = tf.summary.FileWriter(current_dir + '/train', sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(20000):
    train_batch_x, train_batch_y = batch(1000)
    step,summary,acc,loss = sess.run([train_step,merged,accuracy,cross_entropy],feed_dict={x:train_batch_x, y_: train_batch_y, keep_prob: 0.5})
    print(i,acc,loss)
    train_writer.add_summary(summary, i)

train_writer.close()