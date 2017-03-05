# coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x为随机100个小于1大于0的数
x = np.random.rand(100).astype(np.float32)
y = x * 0.2 + 0.3

# 定义权重从 -1.0 到 1.0 随机初始化，
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 定义偏差位0
b = tf.Variable(tf.zeros([1]))
# 输出
y_ = W * x + b

# 定义损失函数为 (y - y_)^2 求和的平均值
loss = tf.reduce_mean(tf.square(y - y_))

# 定义优化器为 梯度下降算法，学习速率为 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)

# 定义训练函数位最小化损失函数
train = optimizer.minimize(loss)

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())
    plt.plot(x, y, "ro")
    plt.axis([0, 1, 0, 1])
    plt.show()
    for step in range(101):
        # 进行训练
        sess.run(train)
        if step % 20 == 0:
            _W = sess.run(W)
            _b = sess.run(b)
            print(step, _W, _b)
            # 最后的结果 _W _b 接近 上面定义的方程
            # plt.plot([0,1],[_W*0+_b,_W*1+_b])
            # plt.axis([0, 1, 0, 1])
            # plt.show()
