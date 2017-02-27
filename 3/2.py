#! coding=utf-8
'''
采用多个隐藏层来解决两个不同分区的问题
'''
import random
import math
import tensorflow as tf
import numpy as np

x_one_len = 20
x_len=x_one_len*2
def batch(batch_size):
    train_x = []
    train_y = []
    for i in range(batch_size):
        x1 = random.randint(0, 500)
        x2 = random.randint(0, 500)
        y = [0, 1]
        if (x1 < 200 and x2 < 200) or (x1 > 300 and x2 > 300):
            y = [1, 0]
        x=np.binary_repr(x1, width=x_one_len)+np.binary_repr(x2, width=x_one_len) #\
        #  +np.binary_repr(x1*x2, width=x_one_len)+np.binary_repr(x1*x1, width=x_one_len)\
        #  +np.binary_repr(x2*x2, width=x_one_len)
        x_=list(map(int, x))
        train_x.append(x_)
        train_y.append(y)
    return np.array(train_x), np.array(train_y)

def add_layer(inputs, in_size, out_size, activity_func=None):
    W = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0))
    b =  tf.Variable(tf.constant(0.1, shape=[out_size]))
    Wx_Plus_b = tf.matmul(inputs, W) + b
    Wx_Plus_b = tf.nn.dropout(Wx_Plus_b, keep_prob)
    if activity_func is None:
        outputs = Wx_Plus_b
    else:
        outputs = activity_func(Wx_Plus_b)
    return outputs
    

# 隐藏层
hidden_units = 255
# 输入维度
n_input = x_len
# 输出维度
n_output = 2
# 学习比例
learning_rate = 0.0001
x = tf.placeholder(tf.float32, [None, n_input])
y_ = tf.placeholder(tf.float32, [None, n_output])
# 随机抛弃率
keep_prob = tf.placeholder(tf.float32)

# 增加隐藏层
l1 = add_layer(x, n_input, hidden_units, activity_func=tf.nn.tanh)
l2 = add_layer(l1, hidden_units, hidden_units, activity_func=tf.nn.tanh)
#l3 = add_layer(l2, hidden_units, hidden_units, activity_func=tf.nn.tanh)
#l4 = add_layer(l3, hidden_units, hidden_units, activity_func=tf.nn.tanh)

#prediction = add_layer(l1, hidden_units, n_output, activity_func=tf.nn.softmax)
#prediction = add_layer(l2, hidden_units, n_output, activity_func=tf.nn.relu)
#prediction = add_layer(l2, hidden_units, n_output, activity_func=tf.nn.tanh)
prediction = add_layer(l2, hidden_units, n_output, activity_func=tf.nn.softmax)
## coss and train step
#cross_entropy = -tf.reduce_sum(y_ * tf.log(prediction))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction)))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# 正确率
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

# 每批执行数量
batch_size = 10000
with tf.Session() as sess:
    sess.run(init)
    for i in range(100000):
        batch_x, batch_y = batch(batch_size)
        _,loss,acc=sess.run([train_op,cross_entropy,accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob:0.5})           
        print(i, acc, loss)


    
