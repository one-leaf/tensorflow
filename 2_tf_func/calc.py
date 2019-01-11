# coding=utf-8
import tensorflow as tf
import numpy as np

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

# 组合
def stack(sess):
    x = tf.constant([1, 4])
    y = tf.constant([2, 5])
    z = tf.constant([3, 6])
    o = tf.stack([x, y, z])
    print(o.get_shape())    # (3,   2)
    print(sess.run(o))  # [[1 4] [2 5] [3 6]]
    o = tf.stack([x, y, z], axis=1)
    print(o.get_shape())    # (2,   3)
    print(sess.run(o))  # [[1 2 3] [4 5 6]]

def stack2():
    x = [[1, 4],[7, 10]]
    y = [[2, 5],[8, 11]]
    z = [[3, 6],[9, 12]]
    print(np.stack((x,y,z)).shape)  # (3, 2, 2)
    print(np.stack((x,y,z),axis=1).shape)   # (2, 3, 2)
    print(np.stack((x,y,z),axis=2).shape)   # (2, 2, 3)
    print(np.stack((x,y,z),axis=2))     # [[[ 1  2  3]  [ 4  5  6]] [[ 7  8  9]  [10 11 12]]]

def matmul(sess):
    a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3]) 
    b = tf.constant([0.1,0.2], shape=[2, 1])
    z = a*b
    print(sess.run(z))
    
if __name__ == '__main__':
    with tf.Session() as sess:
        # stack(sess)
        # stack2()
        matmul(sess)