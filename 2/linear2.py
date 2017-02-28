# coding:utf-8
import tensorflow as tf
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def batch(batch_size):
    train_x = []
    train_y = []
    for i in range(batch_size):
        x = 1.0*random.randint(0, 500)
        y = x*math.tanh(20)+random.randint(-100,100)
        train_x.append(x)
        train_y.append(y)
    return np.array(train_x), np.array(train_y)

# 定义参数，分别是学习率，迭代次数，还有一个是定义每10次迭代打印一些内容
learning_rate = 0.01
train_epochs = 100
display_step = 10


x = tf.placeholder(dtype='float')
y_ = tf.placeholder(dtype='float')

# 定义两个需要求出的w和b变量
W = tf.Variable(np.random.randn())
b = tf.Variable(np.random.randn())

# 预测值
y=tf.multiply(W, x) + b

# 定义代价损失和优化方法
#cost = tf.reduce_mean(tf.square(y - y_))
#n_samples = train_X.shape[0]
n = tf.placeholder(dtype='float')

cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * n)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


init = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(train_epochs):
        train_X,train_Y = batch(1000)
        _n=train_X.shape[0]
        for (_x, _y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={x: _x, y_: _y,n:_n})
        # 每轮打印一些内容
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={x: train_X, y_: train_Y,n:_n})
            print('Epochs:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(c), 'W=', sess.run(W), 'b=', sess.run(b))
    print('optimizer finished')
    train_X,train_Y = batch(1000)
    _n=train_X.shape[0]    
    training_cost = sess.run(cost, feed_dict={x: train_X, y_: train_Y,n:_n})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
   
    plt.plot(train_X, train_Y, 'ro', label='origin data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()   