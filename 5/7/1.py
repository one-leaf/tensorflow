# coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

draw_point_count = 100
radom_data_count = 5

def true_data(batch_size):
    a = np.random.uniform(1,2,size=batch_size)[:,np.newaxis] # a=[[...]]
    PAINT_POINTS = np.vstack([np.linspace(-1, 1, draw_point_count) for _ in range(batch_size)])   # [[15][15]]
    data = a * np.power(PAINT_POINTS, 2)+(a-1)
    return data

with tf.variable_scope("G"):
    g_in = tf.placeholder(tf.float32, [None, radom_data_count])
    g_1 = tf.layers.dense(g_in, 256, tf.nn.relu)
    g_out = tf.layers.dense(g_1, draw_point_count)

with tf.variable_scope("D"):
    true_in = tf.placeholder(tf.float32,[None, draw_point_count])
    d_1     = tf.layers.dense(true_in, 256, tf.nn.relu, name='1')
    prob_1  = tf.layers.dense(d_1, 1, tf.nn.sigmoid, name='out')    # 0~1

    d_2     = tf.layers.dense(g_out, 256, tf.nn.relu, name='1', reuse=True)
    prob_2  = tf.layers.dense(d_2, 1, tf.nn.sigmoid, name='out', reuse=True)    #0~1

d_loss = -tf.reduce_mean(tf.log(prob_1) + tf.log(1-prob_2))     #-00 ~ 0  0~-00  => prob1 -> 1 prob2 -> 0 
g_loss = -tf.reduce_mean(tf.log(prob_2))                        # prob2 -> 1

train_d = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D'))
train_g = tf.train.AdamOptimizer(1e-4).minimize(g_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G'))
# train_d = tf.train.AdamOptimizer(1e-4).minimize(d_loss)
# train_g = tf.train.AdamOptimizer(1e-4).minimize(g_loss)

def main():
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_size = 64
    plt.ion()
    PAINT_POINTS = np.vstack([np.linspace(-1, 1, draw_point_count) for _ in range(batch_size)])   # [[15][15]]

    for step in range(100000):
        _true_data = true_data(batch_size)
        _g_ideas = np.random.randn(batch_size, radom_data_count)
        _feed = {g_in:_g_ideas, true_in: _true_data}
        _, _g_loss,_prob_2=sess.run([train_g, g_loss, prob_2], _feed)
        # print("g:",_g_loss,_prob_2[0])
        sess.run(train_d, _feed)
        _, _g_out, _prob_1, _d_loss = sess.run([train_d, g_out, prob_1, d_loss ], _feed)
        # print("d:",_d_loss, _prob_1[0])
        if step % 200 == 0:  # plotting
            plt.cla()
            plt.plot(PAINT_POINTS[0], _g_out[0], c='#4AD631', lw=3, label='Generated painting',)
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % _prob_1.mean(), fontdict={'size': 15})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -_d_loss, fontdict={'size': 15})
            plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.01)

if __name__ == '__main__':
    main()