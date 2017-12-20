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

def INCEPTIONV2(inputs):
    # mixed_3b => 256
    layer0 = slim.conv2d(inputs, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(inputs, 64, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(inputs, 64, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer3 = slim.avg_pool2d(inputs, [3,3], stride=1, padding="SAME")
    layer3 = slim.conv2d(layer3, 32, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = tf.concat([layer0, layer1, layer2, layer3], 3)   
    # mixed_3c => 320
    layer0 = slim.conv2d(layer, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 64, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer, 64,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer3 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer3 = slim.conv2d(layer3, 64, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = tf.concat([layer0, layer1, layer2, layer3], 3)        
    # mixed_4a => 576
    layer0 = slim.conv2d(layer, 128, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer0 = slim.conv2d(layer0, 160, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 64,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer = tf.concat([layer0, layer1, layer2], 3)   
    # mixed_4b => 576
    layer0 = slim.conv2d(layer, 224, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 64, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer, 96,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer3 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer3 = slim.conv2d(layer3, 128, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = tf.concat([layer0, layer1, layer2, layer3], 3)  
    # mixed_4c => 576
    layer0 = slim.conv2d(layer, 192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 96, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer, 96,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer3 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer3 = slim.conv2d(layer3, 128, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = tf.concat([layer0, layer1, layer2, layer3], 3)  
    # mixed_4d => 576
    layer0 = slim.conv2d(layer, 160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 128, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 160, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer, 128,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 160, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 160, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer3 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer3 = slim.conv2d(layer3, 96, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = tf.concat([layer0, layer1, layer2, layer3], 3)  
    # mixed_4e => 576
    layer0 = slim.conv2d(layer, 96, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 128, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 192, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer, 160,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 192, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 192, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer3 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer3 = slim.conv2d(layer3, 96, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = tf.concat([layer0, layer1, layer2, layer3], 3)
    # mixed_5a => 1024
    layer0 = slim.conv2d(layer, 128, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer0 = slim.conv2d(layer0, 192, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 192,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer = tf.concat([layer0, layer1, layer2], 3)  
    # mixed_5b => 1024
    layer0 = slim.conv2d(layer, 352, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 192, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 320, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer, 160,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 224, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 224, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer3 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer3 = slim.conv2d(layer3, 128, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = tf.concat([layer0, layer1, layer2, layer3], 3)
    # mixed_5c => 1024
    layer0 = slim.conv2d(layer, 352, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer, 192, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer1 = slim.conv2d(layer1, 320, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer, 192,  [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 224, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer2 = slim.conv2d(layer2, 224, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer3 = slim.avg_pool2d(layer,  [3,3], stride=1, padding="SAME")
    layer3 = slim.conv2d(layer3, 128, [1,1], weights_initializer=slim.init_ops.truncated_normal_initializer(0.0, 0.09),  normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = tf.concat([layer0, layer1, layer2, layer3], 3)
    return layer

# 神经网络定义, CNN
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 28*28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')   
    drop_prob = tf.placeholder(tf.float32) 
    x_image = tf.reshape(x, [-1,28,28,1])

    layer = INCEPTIONV2(x_image)

    layer = tf.contrib.layers.flatten(layer)
    prediction = tf.layers.dense(layer, 10)

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
    while mnist.train.epochs_completed < 8:
        batch_x, batch_y= getBatch(8)
        _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y, drop_prob: 0.25})
        if step % 10 == 0 :
            acc = sess.run(accuracy, feed_dict={x: valid_x[:8], y: valid_y[:8], drop_prob: 0})
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

    acc = sess.run(accuracy, feed_dict={x: test_x[:8], y: test_y[:8], drop_prob: 0})
    print("Last accuracy:",acc)
    # Last accuracy: 
