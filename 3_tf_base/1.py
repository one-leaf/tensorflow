# coding=utf-8
'''
最简单的tf
'''
import tensorflow as tf
import numpy as np

import random

x_one_len = 20
x_len=x_one_len*2

def batch(batch_size):
    train_x = []
    train_y = []
    for i in range(batch_size):
        x1 = random.randint(0, 500)
        x2 = random.randint(0, 500)
        y = [0, 1]
        if x1 < 200 and x2 < 200:
            y = [1, 0]
        x=np.binary_repr(x1, width=x_one_len)+np.binary_repr(x2, width=x_one_len)
        x_=list(map(int, x))
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

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for i in range(100):
    batch_xs, batch_ys = batch(10000)
    _,acc,loss=sess.run([train_step,accuracy,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    print(i,acc,loss)

test_x, test_y = batch(10)
_y = y.eval(feed_dict={x: test_x})
for i in range(len(test_x)):
    m = ''.join(str(x) for x in test_x[i])
    x1=int(m[:x_one_len], 2)
    x2=int(m[x_one_len:], 2)
    print(x1, x2, test_y[i], [1,0] if _y[i][0]>_y[i][1] else [0,1], _y[i])
