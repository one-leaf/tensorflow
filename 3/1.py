# coding=utf-8
import tensorflow as tf
import numpy as np

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
        if (x1 < 200 and x2 < 200) or (x1 > 300 and x2 > 300):
            y = [1, 0]
        x=np.binary_repr(x1, width=x_one_len)+np.binary_repr(x2, width=x_one_len)\
          +np.binary_repr(x1*x2, width=x_one_len)+np.binary_repr(x1*x1, width=x_one_len)\
          +np.binary_repr(x2*x2, width=x_one_len)
        x_=map(int, x)
        train_x.append(x_)
        train_y.append(y)
    return np.array(train_x), np.array(train_y)

x = tf.placeholder(tf.float32, [None, x_len])
W = tf.Variable(tf.random_normal([x_len, 2], stddev=0.01))
b = tf.Variable(tf.random_normal([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000000):
    batch_xs, batch_ys = batch(10000)
    _,loss=sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_xs,test_ys = batch(200)
    print(i,loss,sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))

