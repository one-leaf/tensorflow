import tensorflow as tf
import tensorflow.contrib.slim as slim
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

# 增加 DenseNet 网络 # 采用 64 会内存过大
def addDenseNetLayer(inputs, layer_size):
    nodes = []
    layer = slim.conv2d(inputs, 16, [3,3], normalizer_fn = slim.batch_norm) 
    nodes.append(layer)
    for i in range(layer_size):
        layer = tf.concat(nodes, axis=3)
        outputs = slim.conv2d(layer, 16, [3,3],normalizer_fn=slim.batch_norm)
        nodes.append(outputs)
    return outputs    

# 神经网络定义, CNN
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 28*28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')   
    x_image = tf.reshape(x, [-1,28,28,1])

    layer = slim.conv2d(x_image, 64, [3,3], normalizer_fn=slim.batch_norm)
    for i in range(5):
        layer = addDenseNetLayer(layer, 5)
        layer = slim.conv2d(layer, 64, [3,3], stride=[2, 2], normalizer_fn=slim.batch_norm)    

    layer = slim.conv2d(layer, 10, [3,3], normalizer_fn=slim.batch_norm, activation_fn=None)
    prediction = slim.layers.softmax(slim.layers.flatten(layer))

    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, prediction, optimizer, cost, accuracy

if __name__ == '__main__':
    x, y, prediction, optimizer, cost, accuracy = neural_networks()
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
        _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y})
        if step % 10 == 0 :
            acc = sess.run(accuracy, feed_dict={x: valid_x, y: valid_y})
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

    acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
    print("Last accuracy:",acc)
    # Last accuracy: 0.9925
