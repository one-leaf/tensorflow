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
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 增加卷积层
def add_conv_layer(inputs, patch_size, in_size, out_size, norm=False, activation_function=None, pool_function=None):
    Weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([out_size]) + 0.1)
    layer = tf.nn.conv2d(inputs, Weights, strides=[1, 1, 1, 1], padding='SAME')    
    Wx_plus_b = layer + biases

    if norm:
        fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0])
        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
        mean, var = mean_var_with_update()
        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)

    if activation_function is None:
        convlayer = Wx_plus_b
    else:
        convlayer = activation_function(Wx_plus_b)
    if pool_function is None:
        outputs = convlayer
    else:
        outputs = pool_function(convlayer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return outputs

# def batch_norm(x, n_out, phase_train):
#     beta = tf.Variable(tf.constant(0.0, shape=[n_out]))
#     gamma = tf.Variable(tf.constant(1.0, shape=[n_out]))
#     batch_mean, batch_var = tf.nn.moments(x, [0,1,2])
#     ema = tf.train.ExponentialMovingAverage(decay=0.5)

#     def mean_var_with_update():
#         ema_apply_op = ema.apply([batch_mean, batch_var])
#         with tf.control_dependencies([ema_apply_op]):
#             return tf.identity(batch_mean), tf.identity(batch_var)

#     mean, var = tf.cond(phase_train,
#                         mean_var_with_update,
#                         lambda: (ema.average(batch_mean), ema.average(batch_var)))
#     normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed

# 神经网络定义, RNN
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 28*28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y') 
    
    x_image = tf.reshape(x, [-1,28,28,1])
    layer = add_conv_layer(x_image, 5, 1, 32, norm=True, activation_function=tf.nn.relu) 
    layer = add_conv_layer(layer, 3, 32, 64, norm=True, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 
    layer = add_conv_layer(layer, 3, 64, 64, norm=True, activation_function=tf.nn.relu) 
    layer = add_conv_layer(layer, 3, 64, 128, norm=True, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 
    image_width = image_height = 28//2//2
    layer_size = image_width*image_height*128

    layer =  tf.reshape(layer, [-1,layer_size])
    layer = add_layer(layer, layer_size, 128, activation_function=tf.nn.relu)
    layer = tf.reshape(layer, (-1, 1, 128))

    num_units = 128

    cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units//2, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units//2, state_is_tuple=True)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, layer, dtype=tf.float32)
    logits = tf.concat(outputs, axis=2)

    logits = tf.transpose(logits, (0, 2, 1)) 
    # [batch_size, time_step, num_units] = > [batch_size, num_units, time_step] 不转也能学的
    logits = tf.reshape(logits,[-1, num_units])
    prediction = add_layer(logits, num_units, 10)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

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
    # Last accuracy: 0.989
