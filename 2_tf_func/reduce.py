# coding=utf-8

import tensorflow as tf

# sum(x)
def reduce_sum(sess):
    x = tf.constant([[1, 1, 1],[1, 1, 1]])
    y = tf.reduce_sum(x)
    print('x',sess.run(x),'y',sess.run(y))
    # x [[1 1 1] [1 1 1]] y 6
    y = tf.reduce_sum(x, 0) 
    print('x',sess.run(x),'y',sess.run(y))
    # x [[1 1 1] [1 1 1]] y [2 2 2]
    x = tf.constant([[1, 2],[3, 4]])
    y = tf.reduce_sum(x,reduction_indices = 1)
    print('x',sess.run(x),'y',sess.run(y))
    # x [[1 2] [3 4]] y [3, 7]

# *(x)
def reduce_prod(sess):
    x = tf.constant([[2, 2, 2],[2, 2, 2]])
    y = tf.reduce_prod(x)
    print('x',sess.run(x),'y',sess.run(y))
    # x [[2 2 2] [2 2 2]] y 64

# min(x)
def reduce_min(sess):
    x = tf.constant([[1, 2, 3],[1, 2, 3], [0,0,0], [-2,1,-1]])
    y = tf.reduce_min(x, axis=0)
    print('x',sess.run(x),'y',sess.run(y))
    # x [[1 2 3] [1 2 3]] y 1

# max(x)
def reduce_max(sess):
    x = tf.constant([[1, 2, 3],[1, 2, 3],[0,0,0]])
    y = tf.reduce_max(x, axis=-1)
    print('x',sess.run(x),'y',sess.run(y))    
    # x [[1 2 3] [1 2 3]] y 3

# avg(x)
def reduce_mean(sess):
    x = tf.constant([[1, 2, 3],[1, 2, 3]])
    y = tf.reduce_mean(x)
    print('x',sess.run(x),'y',sess.run(y))    
    # x [[1 2 3] [1 2 3]] y 2

# and(x)
def reduce_all(sess):
    x = tf.constant([[True, False, True],[False, True, True]])
    y = tf.reduce_all(x)
    print('x',sess.run(x),'y',sess.run(y))    
    # x [[ True False  True] [False  True  True]] y False

# or(x)
def reduce_any(sess):
    x = tf.constant([[True, False, True],[False, True, True]])
    y = tf.reduce_any(x)
    print('x',sess.run(x),'y',sess.run(y))    
    # x [[ True False  True] [False  True  True]] y True

# log(sum(exp(x)))
def reduce_logsumexp(sess):
    x = tf.constant([[0., 0., 0.],[0., 0., 0.]])
    y = tf.reduce_logsumexp(x)
    print('x',sess.run(x),'y',sess.run(y))      
    # x [[ 0.  0.  0.] [ 0.  0.  0.]] y 1.79176

# count(!0)
def count_nonzero(sess):
    x = tf.constant([[0., 1., 2.],[1., 0., 2.]])
    y = tf.count_nonzero(x)
    print('x',sess.run(x),'y',sess.run(y))  
    # x [[ 0.  1.  2.] [ 1.  0.  2.]] y 4

# x1 + x2
def accumulate_n(sess):
    x = tf.constant([[0, 1, 2],[1, 0, 2]])
    y = tf.accumulate_n([x, x])
    print('x',sess.run(x),'y',sess.run(y))  
    # x [[0 1 2] [1 0 2]] y [[0 2 4] [2 0 4]]

if __name__ == '__main__':
    with tf.Session() as sess:    
        #reduce_sum(sess)
        # reduce_prod(sess)
        reduce_min(sess)
        reduce_max(sess)
        # reduce_mean(sess)
        # reduce_all(sess)
        # reduce_any(sess)
        # reduce_logsumexp(sess)
        # count_nonzero(sess)
        # accumulate_n(sess)