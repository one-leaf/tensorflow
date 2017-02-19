#! coding=utf-8
import random
import math
import tensorflow as tf
import numpy as np

x_one_len = 20
x_len=x_one_len*2

def batch(batch_size):
    train_x = []
    train_y = []
    for i in range(batch_size):
        x1 = random.randint(0, 500)
        x2 = random.randint(0, 500)
        y = [0, 1]
        if (x1 < 200 and x2 < 200) or (x1 > 300 and x2 > 300):
            y = [1, 0]
        x=np.binary_repr(x1, width=x_one_len)+np.binary_repr(x2, width=x_one_len) #\
        #  +np.binary_repr(x1*x2, width=x_one_len)+np.binary_repr(x1*x1, width=x_one_len)\
        #  +np.binary_repr(x2*x2, width=x_one_len)
        x_=list(map(int, x))
        train_x.append(x_)
        train_y.append(y)
    return np.array(train_x), np.array(train_y)

def add_layer(inputs, in_size, out_size, layer_name, activity_func=None):
    W = tf.Variable(tf.random_uniform([in_size, out_size], -1.0, 1.0), name="W")
    bias =  tf.Variable(tf.constant(0.1, shape=[out_size]), name="bias")
    #Wx_Plus_b = tf.matmul(inputs, W) + bias
    Wx_Plus_b = tf.nn.xw_plus_b(inputs, W, bias)
    #Wx_Plus_b = tf.nn.dropout(Wx_Plus_b, keep_prob)
    if activity_func is None:
        outputs = Wx_Plus_b
    else:
        outputs = activity_func(Wx_Plus_b)
    return outputs
    

## para
hidden_layers = 2
hidden_units = 8
n_input = x_len
n_classes = 2
learning_rate = 0.001
## define network
x = tf.placeholder(tf.float32, [None, n_input], name="input")
y_ = tf.placeholder(tf.float32, [None, n_classes], name="output")
keep_prob = tf.placeholder(tf.float32)
l1 = add_layer(x, n_input, hidden_units, 'hidden_layer_1', activity_func=tf.nn.tanh)
l2 = add_layer(l1, hidden_units, hidden_units, 'hidden_layer_2', activity_func=tf.nn.tanh)
#l3 = add_layer(l2, hidden_units, hidden_units, 'hidden_layer_3', activity_func=tf.nn.tanh)
#l4 = add_layer(l3, hidden_units, hidden_units, 'hidden_layer_4', activity_func=tf.nn.tanh)

#prediction = add_layer(l1, hidden_units, n_classes, 'prediction_layer', activity_func=tf.nn.softmax)
#prediction = add_layer(l2, hidden_units, n_classes, 'prediction_layer', activity_func=tf.nn.relu)
prediction = add_layer(l2, hidden_units, n_classes, 'prediction_layer', activity_func=tf.nn.tanh)
#prediction = add_layer(l2, hidden_units, n_classes, 'prediction_layer', activity_func=tf.nn.softmax)
## coss and train step
#cross_entropy = -tf.reduce_sum(y_ * tf.log(prediction))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction)))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
tf.summary.scalar('loss', cross_entropy)
## accuracy
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
# para
batch_size = 10000
with tf.Session() as sess:
#    write = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(init)
    for step in range(100000):
        batch_x, batch_y = batch(batch_size)
        _,_loss,_acc=sess.run([train_op,cross_entropy,accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0})           
        #test_x,test_y=batch(200)   
        #print(_loss)
        print('step', step, "loss:",_loss, 'accuracy:', _acc )
        #print('epoch', epoch, 'accuracy:', sess.run(accuracy, feed_dict={x: test_x, y_: test_y, keep_prob:1.0}),"loss:",_loss)
    #print('*' * 30)
    #test_x,test_y=batch(200)
    #print('training finish. accuracy:', sess.run(accuracy, feed_dict={x: test_x, y_: test_y, keep_prob:1.0}))


    
