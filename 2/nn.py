# coding=utf-8
import tensorflow as tf

# max(x,0)
def relu(sess):
    x = tf.constant([-1.0, 2.0])
    y = tf.nn.relu(x)
    print("relu:","x",sess.run(x),"y",sess.run(y))  
    # relu: x [-1.  2.] y [ 0.  2.]

# min(max(x,0),6)
def relu6(sess):
    x = tf.constant([-1.0, 12.0])
    y = tf.nn.relu6(x)
    print("relu6:","x",sess.run(x),"y",sess.run(y))  
    # relu6: x [ -1.  12.] y [ 0.  6.]

# log(exp(x)+1)
def softplus(sess):
    x = tf.constant([-1.0, 12.0])
    y = tf.nn.softplus(x)
    print("softplus:","x",sess.run(x),"y",sess.run(y))  
    # softplus: x [ -1.  12.] y [  0.31326166  12.00000572]

# 1/(1+exp(-x))
def sigmoid(sess):
    x = tf.constant([[1.0, 2.0],[1.0, 2.0]])
    y = tf.nn.sigmoid(x)
    print("sigmoid:","x",sess.run(x),"y",sess.run(y))  
    # sigmoid: x [[ 1.  2.] [ 1.  2.]] y [[ 0.7310586   0.88079703] [ 0.7310586   0.88079703]]

# (exp(x) - exp(-x)) / (exp(x) + exp(-x))
def tanh(sess):
    x = tf.constant([[1.0, 2.0],[1.0, 2.0]])
    y = tf.nn.tanh(x)
    print("tanh:","x",sess.run(x),"y",sess.run(y))  
    # tanh: x [[ 1.  2.] [ 1.  2.]] y [[ 0.76159418  0.96402758] [ 0.76159418  0.96402758]]


# 按照 0.5 的概率是否放电，不放电设置位 0 否则 *(1/0.5) 倍
def dropout(sess):
    a = tf.constant([-1.0, 2.0, 3.0, 4.0])
    b = tf.nn.dropout(a, 0.5)
    print("dropout:","input",sess.run(a),"output",sess.run(b))      
    # dropout: input [-1.  2.  3.  4.] output [-2.  4.  0.  8.]
    b = tf.nn.dropout(a, 1)
    print("dropout:","input",sess.run(a),"output",sess.run(b))      
    # dropout: input [-1.  2.  3.  4.] output [-1.  2.  3.  4.]

# 卷积运算
def conv2d(sess):
    x = tf.ones([1,4,4,1])  # [数量，图片宽，图片高, 图片通道数]
    f = tf.ones([2,2,1,3])  # [采样块宽，采样块高, 采样通道数, 输出通道数]
    y = tf.nn.conv2d(x,f,[1,1,1,1],padding='SAME')  # 步长都为1
    print(sess.run(y))
    # [[[[ 4.  4.  4.]   [ 4.  4.  4.]   [ 4.  4.  4.]   [ 2.  2.  2.]]
    #   [[ 4.  4.  4.]   [ 4.  4.  4.]   [ 4.  4.  4.]   [ 2.  2.  2.]]
    #   [[ 4.  4.  4.]   [ 4.  4.  4.]   [ 4.  4.  4.]   [ 2.  2.  2.]]
    #   [[ 2.  2.  2.]   [ 2.  2.  2.]   [ 2.  2.  2.]   [ 1.  1.  1.]]]]
    print(y.get_shape())
    # (1, 4, 4, 3)

# 池化, 将数据按大小球平均值缩小，降低计算复杂度
def max_pool(sess):
    x = tf.ones([1,4,4,1])  # [数量，图片宽，图片高, 图片通道数]
    y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(sess.run(y))
    # [[[[ 1.]   [ 1.]]  [[ 1.]   [ 1.]]]]    
    print(y.get_shape())
    # (1, 2, 2, 1)

if __name__ == '__main__':
    with tf.Session() as sess:
        # relu(sess)
        # relu6(sess)
        # softplus(sess)
        # dropout(sess)
        # sigmoid(sess)
        # tanh(sess)
        # conv2d(sess)
        max_pool(sess)