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

# 神经网络定义, RNN
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 28*28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y') 
    
    x_image = tf.reshape(x, [-1,28,28,1])
    layer = add_conv_layer(x_image, 5, 1, 32, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 
    layer = add_conv_layer(layer, 3, 32, 64, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool)     
    layer = add_conv_layer(layer, 3, 64, 128, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool)     
    image_width = image_height = 28//2//2
    layer_size = image_width*image_height
    x_image =  tf.reshape(layer, [-1,layer_size*128])

    layer = add_layer(x_image, layer_size*128, layer_size*64, tf.nn.sigmoid) 
    layer = add_layer(layer, layer_size*64, layer_size*32, tf.nn.sigmoid) 
    layer = add_layer(layer, layer_size*32, layer_size*16, tf.nn.sigmoid) 
    layer = add_layer(layer, layer_size*16, layer_size*8, tf.nn.sigmoid) 

    _layer = add_layer(layer, layer_size*8, layer_size*16, tf.nn.sigmoid) 
    _layer = add_layer(_layer, layer_size*16, layer_size*32, tf.nn.sigmoid) 
    _layer = add_layer(_layer, layer_size*32, layer_size*64, tf.nn.sigmoid) 
    _layer = add_layer(_layer, layer_size*64, layer_size*128, tf.nn.sigmoid) 
    _cost  = tf.reduce_sum(tf.square(x_image - _layer))
    _optimizer = tf.train.AdamOptimizer(0.001).minimize(_cost)

    x_image =  tf.reshape(layer, [-1, layer_size, 8])
    x_image = tf.transpose(x_image, (0, 2, 1)) 

    num_units = 64

    cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units//2, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units//2, state_is_tuple=True)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_image, dtype=tf.float32)
    logits = tf.concat(outputs, axis=2)

    logits = tf.transpose(logits, (0, 2, 1)) 
    # [batch_size, time_step, num_units] = > [batch_size, num_units, time_step] 不转也能学的
    logits = tf.reshape(logits,[-1, 8 * num_units])
    prediction = add_layer(logits, 8 * num_units, 10)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, prediction, optimizer, cost, accuracy, _optimizer, _cost

if __name__ == '__main__':
    x, y, prediction, optimizer, cost, accuracy, _optimizer, _cost = neural_networks()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    valid_x, valid_y = getValidationImages()
    test_x, test_y = getTestImages()

    for i in range(100000):
        batch_x, batch_y= getBatch(100)
        _, loss = sess.run([_optimizer, _cost], feed_dict={x: batch_x})
        if i % 100 == 0 :
            print(i,loss)
    mnist.train._index_in_epoch = 0
    mnist.train._epochs_completed = 0

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
    # Last accuracy: 0.989
