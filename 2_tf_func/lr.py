# coding=utf-8
import tensorflow as tf

# global_step 当前步骤
# starter_learning_rate 开始学习指数
# decay_steps 多少步计算一次
# decay_rate 每次下降的幅度
# staircase 是否向下取整
# decayed_learning_rate = learning_rate * decay_rate ^ t
def exponential_decay():
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    decay_steps = 10
    decay_rate = 0.9
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           decay_steps, decay_rate, staircase=False)
    for i in range(1,101):
        with tf.Session() as sess:
            if i%10==0:
                rate = sess.run(learning_rate,{global_step:i})
                print(i,rate)

def polynomial_decay():
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    decay_steps = 10
    end_learning_rate = 0.0001
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                           decay_steps, end_learning_rate=end_learning_rate, 
                                           power=2, cycle=True)
    for i in range(1,101):
        with tf.Session() as sess:
            rate = sess.run(learning_rate,{global_step:i})
            print(i,rate)
    
# decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
def natural_exp_decay():
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    decay_steps = 10
    decay_rate = 0.9
    learning_rate = tf.train.natural_exp_decay(starter_learning_rate, global_step,
                                           decay_steps, decay_rate, staircase=False)
    for i in range(1,101):
        with tf.Session() as sess:
            if i%10==0:
                rate = sess.run(learning_rate,{global_step:i})
                print(i,rate)    

# decayed_learning_rate = learning_rate / (1 + decay_rate * t)
def inverse_time_decay():
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    decay_steps = 10
    decay_rate = 0.9
    learning_rate = tf.train.inverse_time_decay(starter_learning_rate, global_step, 
                                            decay_steps, decay_rate, staircase=False)

    for i in range(1,101):
        with tf.Session() as sess:
            if i%10==0:
                rate = sess.run(learning_rate,{global_step:i})
                print(i,rate)    

# 指定范围
def piecewise_constant():
    global_step = tf.Variable(0, trainable=False)
    boundaries = [20, 50, 80]
    values = [0.01, 0.006, 0.003, 0.002]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    for i in range(1,101):
        with tf.Session() as sess:
            if i%10==0:
                rate = sess.run(learning_rate,{global_step:i})
                print(i,rate)    

if __name__ == '__main__':
    # exponential_decay()
    # inverse_time_decay()
    piecewise_constant()