# coding=utf-8

from generate_captcha import gen_captcha_text_and_image as captcha
import numpy as np
from utils import img2gray,img2vec,text2vec,vec2text
import tensorflow as tf

image_h=80
image_w=60
char_set="0123456789"
char_size=len(char_set)
train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

#批量参数数据
def get_batch(batch_size=128):
    batch_x = np.zeros([batch_size, image_h*image_w])
    batch_y = np.zeros([batch_size, char_size])
    for i in range(batch_size):
        text, image = captcha(char_set=char_set,captcha_size=1,width=image_w, height=image_h)
        batch_x[i,:] = img2vec(img2gray(image))
        batch_y[i,:] = text2vec(char_set,text)
    return batch_x, batch_y

#输入
x = tf.placeholder("float", shape=[None, image_h*image_w])
#输出
y_ = tf.placeholder("float", shape=[None, char_size])
#权重
W = tf.Variable(tf.zeros([image_h*image_w,char_size]))
#偏置
b = tf.Variable(tf.zeros([char_size]))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#类别预测
y = tf.nn.softmax(tf.matmul(x,W) + b)
#损失函数
loss = -tf.reduce_sum(y_*tf.log(y))
#训练模型
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

step=0
for i in range(10000):
    batch_x, batch_y = get_batch(128)
    _, loss_ = sess.run([optimizer, loss], feed_dict={x: batch_x, y_: batch_y})
    step+=1
    print(step, loss_)
    if step % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        batch_x_test, batch_y_test = get_batch(100)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy.eval(feed_dict={x: batch_x_test, y_: batch_y_test}))
