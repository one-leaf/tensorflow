
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# 获得数据 , batchSize/10 是为了数据好看一点
def getBatch(batchSize):    
    start = random.randint(1, 100000)
    n = np.linspace(start, start+batchSize/10, batchSize, endpoint=True).reshape((batchSize,1))
    x = np.sin(n)
    y = np.cos(n)
    return x,y,n

# 增加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 神经网络定义
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 1], name='x')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    l1 = add_layer(x, 1, 10, tf.nn.relu)
    prediction = add_layer(l1, 10, 1, tf.nn.relu)
    cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)
    return x, y, prediction, optimizer, cost

if __name__ == '__main__':
    x, y, prediction, optimizer, cost = neural_networks()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    plt.ion()
    plt.show()
    for i in range(10000):
        batch_x, batch_y, batch_n= getBatch(100)
        _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y})
        print(i, loss)
        plt.clf()
        plt.plot(batch_n, batch_y, 'r', batch_n, pred, 'b')
        plt.ylim((-1.2, 1.2))        
        plt.draw()
        plt.pause(0.3)
