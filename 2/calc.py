# coding=utf-8
import tensorflow as tf

# 加法
def add(sess):
    x = tf.add(5, 2)  
    print(sess.run(x))  # 7

# 减法
def sub(sess):
    x = tf.subtract(10, 4) 
    print(sess.run(x))  # 6

# 乘法
def multiply(sess):
    x = tf.multiply(2, 5)  
    print(sess.run(x))  # 10

# 除法
def div(sess):
    x = tf.div(10, 2)  
    print(sess.run(x))  # 5
    x = tf.div(11, 3)  
    print(sess.run(x))  # 3

if __name__ == '__main__':
    with tf.Session() as sess:
        div(sess)