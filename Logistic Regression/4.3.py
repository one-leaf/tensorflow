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

# 增加卷积层
def add_conv_layer(inputs, patch_size, in_size, out_size, activation_function=None, pool_function=None):
    Weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    layer = tf.nn.conv2d(inputs, Weights, strides=[1, 1, 1, 1], padding='SAME')
    Wconvlayer_plus_b = layer + biases
    if activation_function is None:
        convlayer = Wconvlayer_plus_b
    else:
        convlayer = activation_function(Wconvlayer_plus_b)
    if pool_function is None:
        outputs = convlayer
    else:
        outputs = pool_function(convlayer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return outputs

# 神经网络定义, CNN
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 28*28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')   
    drop_prob = tf.placeholder(tf.float32) 
    x_image = tf.reshape(x, [-1,28,28,1])

    layer = x_image
    for i in range(2):
        for j in range(3):
            input  = layer
            layer = tf.layers.conv2d(layer, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
            layer = tf.layers.batch_normalization(layer)
            layer = tf.layers.conv2d(layer, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
            layer = tf.layers.batch_normalization(layer)
            layer = input + layer
        layer = tf.layers.max_pooling2d(layer, pool_size=[2,2], strides=2)
    layer = tf.layers.conv2d(layer, filters=10, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    layer = tf.layers.average_pooling2d(layer,pool_size=[7,7],strides=7)
    prediction = tf.contrib.layers.flatten(layer)

    cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
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
    while mnist.train.epochs_completed < 2:
        batch_x, batch_y= getBatch(100)
        _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y, drop_prob: 0.25})
        if step % 10 == 0 :
            acc = sess.run(accuracy, feed_dict={x: valid_x, y: valid_y, drop_prob: 0})
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

    acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, drop_prob: 0})
    print("Last accuracy:",acc)
    # Last accuracy: 0.9921
