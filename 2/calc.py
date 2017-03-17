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

def append():
    x = np.array([[1,2,3],[4,5,6]]) # (2, 3)
    y = np.stack((x, x, x, x), axis = 2)    # (2, 3, 4) [[[1 1 1 1]  [2 2 2 2]  [3 3 3 3]] [[4 4 4 4]  [5 5 5 5]  [6 6 6 6]]]
    z = np.array( [[[10],[20],[30]],[[40],[50],[60]]] ) # (2, 3, 1)  
    o = np.append(z, y[:,:,0:3 ], axis = 2 )    # [[[10  1  1  1]  [20  2  2  2]  [30  3  3  3]] [[40  4  4  4]  [50  5  5  5]  [60  6  6  6]]]
    p = np.append(z, o[:,:,0:3 ], axis = 2 )    # [[[10 10  1  1]  [20 20  2  2]  [30 30  3  3]] [[40 40  4  4]  [50 50  5  5]  [60 60  6  6]]]
    print(p)

def max():
    x = [1,5,3]
    print(np.max(x))    # 5

if __name__ == '__main__':
    with tf.Session() as sess:
        # stack(sess)
        max()