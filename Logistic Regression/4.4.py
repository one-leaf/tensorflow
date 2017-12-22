import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow.contrib.slim as slim

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

def resNetBlockV1(inputs, size=64):
    layer = slim.conv2d(inputs, size, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer,  size, [3,3], normalizer_fn=slim.batch_norm, activation_fn=None)
    return tf.nn.relu(inputs + layer) 

# 第二种残差模型
def resNetBlockV2(inputs, size=64):
    layer = slim.conv2d(inputs, size,   [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer,  size,   [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer,  size*2, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    return tf.nn.relu(inputs + layer)


def resNet34(layer, isPoolSize=True):
    if isPoolSize:
        stride = 2
        padding = "VALID"
    else:
        stride = 1
        padding = "SAME"
    with slim.arg_scope([slim.max_pool2d, slim.avg_pool2d], stride=stride, padding=padding):
        layer = slim.conv2d(layer, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 

        for i in range(3):
            layer = resNetBlockV1(layer, 64)
        layer = slim.avg_pool2d(layer, [2, 2])

        layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        for i in range(4):
            layer = resNetBlockV1(layer, 128)
        layer = slim.avg_pool2d(layer, [2, 2])

        layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)        
        for i in range(6):
            layer = resNetBlockV1(layer, 256)
        layer = slim.avg_pool2d(layer, [2, 2])

        layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        for i in range(3):
            layer = resNetBlockV1(layer, 512)
        return layer

def resNet50V3(layer, isPoolSize=True):
    if isPoolSize:
        stride = 2
        padding = "VALID"
    else:
        stride = 1
        padding = "SAME"
    with slim.arg_scope([slim.max_pool2d, slim.avg_pool2d], stride=stride, padding=padding):
        layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        
        for i in range(3):
            layer = resNetBlockV2(layer, 64)
        layer = slim.avg_pool2d(layer, [2, 2])

        layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
        for i in range(4):
            layer = resNetBlockV2(layer, 128)
        layer = slim.avg_pool2d(layer, [2, 2])
        half_layer = layer

        layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)        
        for i in range(6):
            layer = resNetBlockV2(layer, 256)
        layer = slim.avg_pool2d(layer, [2, 2])

        layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        for i in range(3):
            layer = resNetBlockV2(layer, 512)
        return layer, half_layer


# 神经网络定义, CNN
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 28*28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')   
    drop_prob = tf.placeholder(tf.float32) 
    x_image = tf.reshape(x, [-1,28,28,1])

    layer, _ = resNet34(x_image)
    layer = slim.avg_pool2d(layer, [3,3])
    layer = tf.contrib.layers.flatten(layer)
    # print(layer.shape)
    prediction = slim.fully_connected(layer,10)

    cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, drop_prob, prediction, optimizer, cost, accuracy

if __name__ == '__main__':
    x, y, drop_prob, prediction, optimizer, cost, accuracy = neural_networks()
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
    while mnist.train.epochs_completed < 1:
        batch_x, batch_y= getBatch(128)
        _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y, drop_prob: 0.25})
        if step % 10 == 0 :
            acc = 0
            key = random.sample(range(len(valid_x)), 64)
            _x =[]
            _y =[]
            for i in key:
                _x.append(valid_x[i])
                _y.append(valid_y[i])
            acc = sess.run(accuracy, feed_dict={x: _x , y: _y , drop_prob: 0})
            acc = acc / 64.
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

    key = random.sample(range(len(test_x)), 64)
    _x =[]
    _y =[]
    for i in key:
        _x.append(test_x[i])
        _y.append(test_y[i])
    acc = sess.run(accuracy, feed_dict={x: _x , y: _y , drop_prob: 0})
    acc = acc / 64.

    print("Last accuracy:",acc)
    # Last accuracy: 
