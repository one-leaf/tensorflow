# coding=utf-8
import tensorflow as tf
import numpy as np

# 新建一个 graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# 新建session with log_device_placement并设置为True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 运行这个 op.
print(sess.run(c))


# 新建一个 graph.
with tf.device('/gpu:1'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# 新建 session with log_device_placement 并设置为 True.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# 运行这个 op.
print(sess.run(c))

# 新建一个 graph.
c = []
for d in ['/gpu:0', '/gpu:1']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# 新建session with log_device_placement并设置为True.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# 运行这个op.
print(sess.run(sum))


# 计算圆周率
c = []
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    n = 1000000
    for d in ['/gpu:0', '/gpu:1']:        
        i=tf.constant(np.random.uniform(size=2*n), shape=[n,2])         # 5000个2维随机数    
        distances=tf.reduce_sum(tf.pow(i,2),1)                          # 随机数的x,y平方和
        tempsum = sess.run(tf.reduce_sum(tf.cast(tf.greater_equal(tf.cast(1.0,tf.float64),distances),tf.float64)))  #是否大于或等于1的个数
        c.append(tempsum)
    with tf.device('/cpu:0'):
         sum = tf.add_n(c)
         print(sess.run(sum/(n*2))*4.0)
