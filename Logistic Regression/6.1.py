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

# 神经网络定义, RNN
def neural_networks():
    x = tf.placeholder(tf.float32, [None, 28*28], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')   
    keep_prob = tf.placeholder(tf.float32) 

    # x_image = tf.reshape(x,[-1,28,28])
    # # x_image = tf.transpose(x_image, (1, 0, 2)) # [time_step, batch_size, input_size]
    # #x_image_shape  = tf.shape(x_image)
    # time_step = 28
    # batch_size = 100
    # input_size = 28
    # x_image = tf.reshape(x_image, [-1, input_size])
    # x_image_shape  = tf.shape(x_image)
    layer = add_layer(x, 28*28, 28*28, activation_function=tf.nn.relu)
    layer = tf.minimum(layer, 20.0)
    layer = tf.nn.dropout(layer, keep_prob)

    # layer = add_layer(layer, 256, 512 , activation_function=tf.nn.relu)
    # layer = tf.minimum(layer, 20.0)    
    # layer = tf.nn.dropout(layer, keep_prob)

    # layer = add_layer(layer, 512, 28*28 , activation_function=tf.nn.relu)
    # layer = tf.minimum(layer, 20.0)    
    # layer = tf.nn.dropout(layer, keep_prob)  # [time_step, 2800]

    x_image = tf.reshape(layer, [-1, 28, 28]) #[-1, time_step , input_size]

    # x_image = tf.transpose(x_image, (1, 0, 2))
    num_units = 64

    cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units//2, state_is_tuple=True)
    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units//2, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_image, dtype=tf.float32)
    logits = tf.concat(outputs, axis=2)
    logits = tf.transpose(logits, (0, 2, 1)) 
    # [batch_size, time_step, num_units] = > [batch_size, num_units, time_step] 不转也能学的
    logits = tf.reshape(logits,[-1, 28 * num_units])

    layer = add_layer(logits, 28 * num_units, 512, activation_function=tf.nn.relu)
    layer = tf.minimum(layer, 20.0)    
    layer = tf.nn.dropout(layer, keep_prob)

    layer = add_layer(layer, 512, 512, activation_function=tf.nn.relu)
    layer = tf.minimum(layer, 20.0)    
    layer = tf.nn.dropout(layer, keep_prob)

    prediction = add_layer(layer, 512, 10)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return x, y, keep_prob, prediction, optimizer, cost, accuracy

if __name__ == '__main__':
    x, y, keep_prob, prediction, optimizer, cost, accuracy = neural_networks()
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
        _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        if step % 10 == 0 :
            acc = sess.run(accuracy, feed_dict={x: valid_x, y: valid_y, keep_prob: 1})
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

    acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1})
    print("Last accuracy:",acc)
    # Last accuracy: 0.977
