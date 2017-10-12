# coding=utf-8
import tensorflow as tf


def shape(sess):
    x = tf.ones([2, 3])
    y = tf.shape(x)
    print("x", sess.run(x), "y", sess.run(y))
    # ('x', array([[ 1.,  1.,  1.],  [ 1.,  1.,  1.]], dtype=float32), 'y', array([2, 3], dtype=int32))

def expand_dims(sess):
    x = tf.ones([2, 3])   # [[1. 1.] [1. 1.] [1. 1.]]
    y = tf.expand_dims(x, 0)
    print(sess.run(y))    # [[[ 1.  1.  1.]  [ 1.  1.  1.]]]
    y = tf.expand_dims(x, 1)
    print(sess.run(y))    # [[[ 1.  1.  1.]] [[ 1.  1.  1.]]]

# 合并数据 以前的函数名为 pack
def stack(sess):
    x1 = [[1, 2, 3], [4, 5, 6]]
    x2 = [[2, 3, 4], [7, 8, 9]]
    y = tf.stack([x1, x2])
    print(sess.run(y))  # [[[1 2 3]  [4 5 6]] [[2 3 4] [7 8 9]]]
    y = tf.stack([x1, x2], axis=1)  # [[[1 2 3] [2 3 4]] [[4 5 6] [7 8 9]]]
    print(sess.run(y))

# 合并数据
def concat(sess):
    x1 = [[1, 2, 3], [4, 5, 6]]
    x2 = [[2, 3, 4], [7, 8, 9]]
    
    y = tf.concat([x1, x2], 0)
    print(y.shape)
    print(sess.run(y))      # [[1 2 3] [4 5 6] [2 3 4] [7 8 9]]
    y = tf.concat([x1, x2], 1) 
    print(sess.run(y))      # [[1 2 3 2 3 4] [4 5 6 7 8 9]]

# 矩阵变形
def reshape(sess):
    x = [[1,2],[3,4],[5,6]]
    y = tf.reshape(x,[2,3])
    print(sess.run(y))  # [[1 2 3] [4 5 6]]
    y = tf.reshape(x,[-1])
    print(sess.run(y))  # [1 2 3 4 5 6]

# 数据抽取
def slice(sess):
    x = [[1,2,3],[4,5,6]]
    y = tf.slice(x,[0,1],[-1,2])
    # [0,1] 第一个0决定了从x的第1行[1,2,3]开始，第二个1，决定了从[1,2,3] 中的2开始抽取
    # [-1,2] 第一个-1决定了抽取开始以下的所有行，在抽取行中抽取2个元素
    print(sess.run(y))  # [[2 3] [5 6]]

if __name__ == '__main__':
    with tf.Session() as sess:
        # shape(sess)
        # expand_dims(sess)
        # stack(sess)
        concat(sess)
        # reshape(sess)
        # slice(sess)
