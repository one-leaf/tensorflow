
# 线性回归

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# 获得数据 / 10 只是让数据好看一点
def getBatch(batchSize,start=None):
    if start==None:    
        start = random.randint(1, 10000)
    n = np.linspace(start, start+batchSize, batchSize, endpoint=True).reshape((batchSize,1)) / 10
    x = np.sin(n)
    y = np.cos(n)
    return x,y,n

# 增加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 神经网络定义
def neural_networks(batch_size, seq_len, cell_size):
    x = tf.placeholder(tf.float32, [None, seq_len, 1], name='x')
    y = tf.placeholder(tf.float32, [None, seq_len, 1], name='y')
    
    _x = tf.reshape(x,[-1,1])
    layer = add_layer(_x, 1, cell_size)
    inputs = tf.reshape(layer,[-1, seq_len, cell_size])

    cell = tf.contrib.rnn.LSTMCell(cell_size, state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    _outputs = tf.reshape(outputs, [-1, cell_size])
    
    prediction = add_layer(_outputs, cell_size, 1, activation_function=tf.nn.tanh)    
    cost = tf.square(tf.subtract(tf.reshape(prediction, [-1]), tf.reshape(y, [-1])))   
    cost = tf.reduce_sum(cost)

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    return x, y, prediction, optimizer, cost

if __name__ == '__main__':
    seq_len = 20
    cell_size = 10
    batch_size = 10
    x, y, prediction, optimizer, cost = neural_networks(batch_size, seq_len, cell_size)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    plt.ion()
    plt.show()
    for i in range(100000):
      #  batch_x, batch_y, batch_n= getBatch(batch_size * seq_len, 0)
        batch_x, batch_y, batch_n= getBatch(batch_size * seq_len)
        batch_x = np.reshape(batch_x,[batch_size, seq_len, 1])
        batch_y = np.reshape(batch_y,[batch_size, seq_len, 1])
        _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y})
        if i % 50 == 0:
            print(i, loss)
            plt.clf()
            plt.plot(batch_n, batch_y.flatten(), 'r', batch_n, pred.flatten(), 'b')
            plt.ylim((-1.2, 1.2))        
            plt.draw()
            plt.pause(0.1)
