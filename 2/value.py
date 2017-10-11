# coding=utf-8
import tensorflow as tf
import numpy as np

def zeros(sess):
    y = np.zeros([2, 3])
    print(y)
    # [[ 0.  0.  0.] [ 0.  0.  0.]]
    y = tf.zeros([2, 3])
    print(sess.run(y))
    # [[ 0.  0.  0.] [ 0.  0.  0.]]
    y = tf.ones_like(y)
    print(sess.run(y))
    # [[ 1.  1.  1.] [ 1.  1.  1.]]

def ones(sess):
    y = tf.ones([2,3])
    print(sess.run(y))
    # [[ 1.  1.  1.] [ 1.  1.  1.]]
    y  = tf.zeros_like(y)
    print(sess.run(y))
    # [[ 0.  0.  0.] [ 0.  0.  0.]]

def fill(sess):
    y = tf.fill([2,3],-1)
    print(sess.run(y))
    # [[-1 -1 -1] [-1 -1 -1]]

# 正太分布随机数，均值mean,标准差stddev
def random_normal(sess):
    y = tf.random_normal([1,5],mean=1,stddev=0.5)
    print(sess.run(y))
    # [[ 0.50446427  0.92098403  0.10005826  0.39223284  1.18514192]]

# 截断正态分布随机数 均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
def truncated_normal(sess):
    y = tf.truncated_normal([1,5],mean=1,stddev=0.5)
    print(sess.run(y))
    # [[ 1.6490171   1.31581068  1.07311356  0.98809385  1.35108364]]

# 从最小值到最大值的均匀分布随机数
def random_uniform(sess):
    y = tf.random_uniform([1,5],2,10)
    print(sess.run(y))
    # [[ 3.98193932  6.87557793  4.12505054  9.27368355  9.27550983]]

# 随机按第一维度打乱顺序
def random_shuffle(sess):
    x = [[1,2],[3,4],[5,6]]
    y = tf.random_shuffle(x)
    print(sess.run(y))
    # [[1 2] [5 6] [3 4]]

# value中的值逐个存入。不够的部分，则全部存入value的最后一个值
def constant(sess):
    y = tf.constant(2,shape=[2,3])
    print(sess.run(y))
    # [[2 2 2] [2 2 2]]
    y = tf.constant([1,2,3],shape=[2,3])
    print(sess.run(y))
    # [[1 2 3] [3 3 3]]

# 获取最大值的位置
def argmax(sess):
    x = [[1,2],[3,2],[4,4]]
    y = tf.argmax(x,0)
    print(sess.run(y))  # [2 2]
    y = tf.argmax(x,1)
    print(sess.run(y))  # [1 0 0]

# 获取最小值的位置
def argmin(sess):
    x = [[1,2],[3,2],[4,4]]
    y = tf.argmin(x,0)
    print(sess.run(y))  # [0 0]
    y = tf.argmin(x,1)
    print(sess.run(y))  # [0 1 0]

# 判断是否相等
def equal(sess):
    x1 = [[1,1],[2,2]]
    x2 = [[1,1],[2,3]]
    y = tf.equal(x1,x2)
    print(sess.run(y))  # [[ True  True] [ True False]]

# 格式转化，常用于boolean到int，配合equal函数使用，通过求平均值来统计正确率
def cast(sess):
    x = [[True, True], [False, True]]
    y = tf.cast(x,"float")
    print(sess.run(y)) # [[ 1.  1.] [ 0.  1.]]
    y = tf.reduce_mean(y) # 0.75
    print(sess.run(y)) 

# 矩阵乘法 [[1*1+2*3 1*2+2*4] [3*1+4*3 3*2+4*4]
def matmul(sess):
    x1 = [[1,2],[3,4]]
    x2 = [[1,2],[3,4]]
    y = tf.matmul(x1,x2)
    print(sess.run(y))  # [[7 10] [15 22]]

# 矩阵反向
def reverse(sess):
    x = tf.random_normal([4,2,3],mean=1,stddev=0.5)
    y = tf.reverse(x,axis=[1])
    # print(sess.run(x))
    print(sess.run([x,y]))

if __name__ == '__main__':
    with tf.Session() as sess:
        # zeros(sess)
        # ones(sess)
        # fill(sess)
        # random_normal(sess)
        # truncated_normal(sess)
        # random_uniform(sess)
        # random_shuffle(sess)
        # constant(sess)
        # argmax(sess)
        # argmin(sess)
        # equal(sess)
        # cast(sess)
        # matmul(sess)
        reverse(sess)