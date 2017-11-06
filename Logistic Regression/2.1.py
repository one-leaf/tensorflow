import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

curr_dir = os.path.dirname(__file__)
mnist = input_data.read_data_sets(os.path.join(curr_dir,"data"), one_hot=True)

# 55000 组图片和标签, 用于训练
def getBatch(batchSize):
    batch_x, batch_y = mnist.train.next_batch(batchSize)
    return batch_x, batch_y  

# 5000 组图片和标签, 用于迭代验证训练的准确性
def getValidationImages():
    return mnist.validation.images, mnist.validation.labels	

# 10000 组图片和标签, 用于最终测试训练的准确性
def getTestImages():
    return mnist.test.images, mnist.test.labels

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
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 28*28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')  
    keep_prob = tf.placeholder(tf.float32) 
    layer = add_layer(x, 28*28, 256, tf.nn.tanh) 
    layer = add_layer(layer, 256, 128, tf.nn.tanh) 

    # 加入自编码，提取特征用，可以降低运算量
    _layer = add_layer(layer, 128, 256, tf.nn.tanh) 
    _layer = add_layer(_layer, 256, 28*28, tf.nn.tanh) 
    _cost  = tf.reduce_sum(tf.square(x - _layer))
    _optimizer = tf.train.AdamOptimizer(0.01).minimize(_cost)

    layer_drop = tf.nn.dropout(layer, keep_prob)    # 一般网络不深，dropout 会没有什么用
    prediction = add_layer(layer_drop, 128, 10) 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, keep_prob, prediction, optimizer, cost, accuracy, _optimizer

if __name__ == '__main__':
    x, y, keep_prob, prediction, optimizer, cost, accuracy, _optimizer = neural_networks()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    valid_x, valid_y = getValidationImages()
    test_x, test_y = getTestImages()

    plt.ion()
    plt.show()
    plt_n=[]
    plt_loss=[]
    plt_acc=[]

    step = 0
    while mnist.train.epochs_completed < 8:
        batch_x, batch_y= getBatch(100)
        _, _, loss, pred = sess.run([_optimizer, optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        if step % 10 == 0 :
            acc = sess.run(accuracy, feed_dict={x: valid_x, y: valid_y, keep_prob: 1.0})
            print(step, loss, acc)
            plt.clf()
            plt_n.append(step)
            plt_loss.append(loss)
            plt_acc.append(acc)
            plt.plot(plt_n, plt_loss, 'b', label="loss")
            plt.plot(plt_n, plt_acc, 'r', label="acc")
            plt.legend(loc='upper right')    
            plt.draw()
            plt.pause(0.1)
        step += 1

    acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
    print("Last accuracy:",acc)
    # Last accuracy: 0.9256
