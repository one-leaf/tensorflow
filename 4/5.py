# coding=utf-8
'''
利用RNN进行机器学习，结果不理想，需要去查原因
'''

from generate_captcha import gen_captcha_text_and_image as captcha
import numpy as np
from utils import img2gray,img2vec,text2vec,vec2text
import tensorflow as tf

image_h=80
image_w=60
image_size=image_h*image_w
char_set="0123456789"
char_size=len(char_set)

# 批量验证码数据
def get_batch(batch_size=128):
    batch_x = np.zeros([batch_size, image_size])
    batch_y = np.zeros([batch_size, char_size])
    for i in range(batch_size):
        text, image = captcha(char_set=char_set,captcha_size=1,width=image_w, height=image_h)
        batch_x[i,:] = img2vec(img2gray(image))
        batch_y[i,:] = text2vec(char_set,text)
    return batch_x, batch_y


rnn_size = 256

x = tf.placeholder('float', [None, image_size]) 
y_ = tf.placeholder('float')
# 定义待训练的神经网络
def recurrent_neural_network(data):
    layer = {'w_':tf.Variable(tf.random_normal([rnn_size, char_size])), 'b_':tf.Variable(tf.random_normal([char_size]))}
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    data = tf.split(data, 1, axis=0)
    outputs, status = tf.contrib.rnn.static_rnn(lstm_cell, data, dtype=tf.float32)
    ouput = tf.add(tf.matmul(outputs[-1], layer['w_']), layer['b_'])
    return ouput

# 使用数据训练神经网络
def train_neural_network():
    predict = recurrent_neural_network(x)
    cost_part = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y_) 
    cost_func = tf.reduce_mean(cost_part)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost_func)
    correct = tf.equal(tf.argmax(predict,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for step in range(1000):
            batch_x, batch_y = get_batch(100)
            _, acc, loss = session.run([optimizer, accuracy, cost_func], feed_dict={x:batch_x,y_:batch_y})
            print(step, acc, loss)
      
train_neural_network()    