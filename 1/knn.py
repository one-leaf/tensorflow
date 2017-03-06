# coding=utf-8

import numpy as np
import tensorflow as tf

import random


def batch(batch_size):
    train_x = []
    train_y = []
    for i in range(batch_size):
        x1 = random.randint(0, 500)
        x2 = random.randint(0, 500)
        y = [0, 1]
        if (x1 < 200 and x2 < 200) or (x1 > 300 and x2 > 300):
            y = [1, 0]
        train_x.append([x1, x2])
        train_y.append(y)
    return np.array(train_x), np.array(train_y)

# 这里从数据集中抽取50000个样本数据作为训练集，200个样本做为测试集,这里train_y是一个2维ndarray
train_x, train_y = batch(50000)
test_x, test_y = batch(200)

# tf Graph的输入
x = tf.placeholder("float", [None, 2])
y = tf.placeholder("float", [2])

# 使用L1距离计算最近邻
# tf.neg()给y中的每个元素取反，这样可以使用tf.add()
# 这里reduction_indices=1是计算每一行的和
distance = tf.reduce_sum(tf.abs(tf.add(x, tf.neg(y))), reduction_indices=1)
# 从distance tensor中寻找最小的距离索引
pred = tf.argmin(distance, 0)

acc = 0.

# 初始化所有 variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # 对每个测试样本，计算它的分类类别
    for i in range(len(test_x)):
        # 获得最近邻
        nn_index = sess.run(pred, feed_dict={x: train_x, y: test_x[i, :]})
        print "nn_indext:", nn_index
        print "train_y[nn_index]:", train_y[nn_index]
        print "np.argmax(train_y[nn_index]):", np.argmax(train_y[nn_index])
        # 获得测试样本的最近邻类别，并将它与真实类别做比较
        print "Test", i, "Prediction:", np.argmax(train_y[nn_index]), \
            "True Class:", np.argmax(test_y[i])
        # 计算 acc
        if np.argmax(train_y[nn_index]) == np.argmax(test_y[i]):
            acc += 1. / len(test_x)
    print "Accuracy:", acc
