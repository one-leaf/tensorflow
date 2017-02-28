# coding=utf-8
import tensorflow as tf

# max(x,0)
def relu():
    x = tf.constant([-1.0, 2.0])
    with tf.Session() as sess:
        y = tf.nn.relu(x)
        print("relu:","x",sess.run(x),"y",sess.run(y))  
        # relu: x [-1.  2.] y [ 0.  2.]

# min(max(x,0),6)
def relu6():
    x = tf.constant([-1.0, 12.0])
    with tf.Session() as sess:
        y = tf.nn.relu6(x)
        print("relu6:","x",sess.run(x),"y",sess.run(y))  
        # relu6: x [ -1.  12.] y [ 0.  6.]

# log(exp(x)+1)
def softplus():
    x = tf.constant([-1.0, 12.0])
    with tf.Session() as sess:
        y = tf.nn.softplus(x)
        print("softplus:","x",sess.run(x),"y",sess.run(y))  
        # softplus: x [ -1.  12.] y [  0.31326166  12.00000572]

# 1/(1+exp(-x))
def sigmoid():
    x = tf.constant([[1.0, 2.0],[1.0, 2.0]])
    with tf.Session() as sess:
        y = tf.nn.sigmoid(x)
        print("sigmoid:","x",sess.run(x),"y",sess.run(y))  
        # sigmoid: x [[ 1.  2.] [ 1.  2.]] y [[ 0.7310586   0.88079703] [ 0.7310586   0.88079703]]

# (exp(x) - exp(-x)) / (exp(x) + exp(-x))
def tanh():
    x = tf.constant([[1.0, 2.0],[1.0, 2.0]])
    with tf.Session() as sess:
        y = tf.nn.tanh(x)
        print("tanh:","x",sess.run(x),"y",sess.run(y))  
        # tanh: x [[ 1.  2.] [ 1.  2.]] y [[ 0.76159418  0.96402758] [ 0.76159418  0.96402758]]


# 按照 0.5 的概率是否放电，不放电设置位 0 否则 *(1/0.5) 倍
def dropout():
    a = tf.constant([-1.0, 2.0, 3.0, 4.0])
    with tf.Session() as sess:
        b = tf.nn.dropout(a, 0.5)
        print("dropout:","input",sess.run(a),"output",sess.run(b))      
        # dropout: input [-1.  2.  3.  4.] output [-2.  4.  0.  8.]
        b = tf.nn.dropout(a, 1)
        print("dropout:","input",sess.run(a),"output",sess.run(b))      
        # dropout: input [-1.  2.  3.  4.] output [-1.  2.  3.  4.]

if __name__ == '__main__':
    # relu()
    # relu6()
    # softplus()
    # dropout()
    # sigmoid()
    tanh()