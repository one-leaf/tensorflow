# coding=utf-8
# 卷积多层多项验证码识别

from generate_captcha import gen_captcha_text_and_image as captcha
import numpy as np
from utils import img2gray,img2vec,text2vec,vec2text
import tensorflow as tf

image_h=80
image_w=200
image_size=image_h*image_w
char_set="0123456789"
char_size=len(char_set)
captcha_size = 4
batch_size = 50

# 批量验证码数据
def get_batch(batch_size=128):
    batch_x = np.zeros([batch_size, image_size])
    batch_y = np.zeros([batch_size, char_size*captcha_size])
    for i in range(batch_size):
        text, image = captcha(char_set=char_set,captcha_size=captcha_size,width=image_w, height=image_h)
        batch_x[i,:] = img2vec(img2gray(image))
        batch_y[i,:] = text2vec(char_set,text)
    return batch_x, batch_y

# 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，
# 第一维 -1 是不限个和 None 类似， 第2、3维对应图片的宽和高，最后一维对应颜色通道的数目，这里是黑白，所以为 1 ，如果图片为 RGB 则为3 。
x = tf.placeholder(tf.float32, [None, image_size])
x_ = tf.reshape(x, [batch_size, image_w, image_h, 1])
y_ = tf.placeholder(tf.float32, [batch_size, char_size*captcha_size])

#卷积层
filter_sizes=[3, 3, 3, 3]
filter_nums=[32,32,32,32]
pool_types=['max','avg','avg','avg']
pool_ksizes=[2,2,2,2]
pool_strides=[1,1,1,1]
conv_pools=[]
for i in range(len(filter_sizes)):
    with tf.variable_scope('conv-pool-{}'.format(i)):    
        if i==0: 
            input = x_
        else:
            input = conv_pools[-1]
        filter_shape=[filter_sizes[i],filter_sizes[i],int(input.get_shape()[-1]),filter_nums[i]]
        W = tf.get_variable("filter", filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable('bias', [filter_nums[i]], initializer=tf.constant_initializer(0.0))
        W_conv = tf.nn.conv2d(input, W, strides=[1, pool_strides[i], pool_strides[i], 1],  padding='VALID')
        conv = tf.nn.relu(tf.nn.bias_add(W_conv, b))
        if pool_types[i]=='avg':  
            pool = tf.nn.avg_pool(conv, ksize=[1, pool_ksizes[i], pool_ksizes[i], 1], strides=[1, pool_strides[i], pool_strides[i], 1], padding='VALID') 
        else:
            pool = tf.nn.max_pool(conv, ksize=[1, pool_ksizes[i], pool_ksizes[i], 1], strides=[1, pool_strides[i], pool_strides[i], 1], padding='VALID') 
        conv_pools.append(pool)

# 全连接层
hidden_sizes = [256]
full_connects = []
for i in range(len(hidden_sizes)):
    with tf.variable_scope('full-connect-{}'.format(i)):
        if i==0:
            batch_size = int(x_.get_shape()[0])
            inputs = tf.reshape(conv_pools[-1],[batch_size,-1])
            in_size = int(inputs.get_shape()[-1])
        else:
            inputs = full_connects[-1]
            in_size = hidden_sizes[i-1]     
        W = tf.get_variable("weights", [in_size,hidden_sizes[i]], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("biases", [hidden_sizes[i]], initializer=tf.constant_initializer(0.0))
        full_connect = tf.nn.relu(tf.matmul(inputs, W) + b)
        full_connects.append(full_connect)

# 由于是多位验证码，继续添加每一位验证码的输出
outputs=[]
for i in range(captcha_size):
    with tf.variable_scope('output-part-{}'.format(i)):
        W = tf.get_variable("weights", [hidden_sizes[-1],char_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("biases", [char_size], initializer=tf.constant_initializer(0.0))
        #fc_part = tf.nn.relu(tf.matmul(full_connects[-1], W) + b)
        fc_part = tf.matmul(full_connects[-1], W) + b
        outputs.append(fc_part)

# 最终输出
output =  tf.concat(outputs, 1)     

# 抛弃函数
losses=[]
for i in range(captcha_size):
    with tf.variable_scope('loss-part-{}'.format(i)):
        outputs_part = tf.slice(output,begin=[0,i*char_size],size=[-1,char_size])
        targets_part = tf.slice(y_,begin=[0,i*char_size],size=[-1,char_size])
        loss_part = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs_part, labels=targets_part)
        reduced_loss_part = tf.reduce_mean(loss_part)
        losses.append(reduced_loss_part)
loss = tf.reduce_mean(losses)

predictions=[]
for i in range(captcha_size):
    with tf.variable_scope('predictions-part-{}'.format(i)):
        outputs_part = tf.slice(output,begin=[0,i*char_size],size=[-1,char_size])
        prediction_part = tf.argmax(outputs_part,axis=1)
        prediction_part = tf.cast(prediction_part, tf.float32)
        predictions.append(prediction_part)
prediction = tf.stack(predictions, axis=1)

predictions_y=[]
for i in range(captcha_size):
    with tf.variable_scope('predictions-y-part-{}'.format(i)):
        outputs_part_y = tf.slice(y_,begin=[0,i*char_size],size=[-1,char_size])
        prediction_part_y = tf.argmax(outputs_part_y,axis=1)
        prediction_part_y = tf.cast(prediction_part_y, tf.float32)
        predictions_y.append(prediction_part_y)
prediction_y = tf.stack(predictions_y, axis=1)

correct_prediction = tf.cast(tf.equal(prediction,prediction_y), tf.float32)
correct_prediction = tf.reduce_mean(correct_prediction, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(correct_prediction, 1.0), tf.float32))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step, 2000, 0.96, staircase=True)

train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(loss,global_step=global_step)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(100000):
    batch = get_batch(batch_size)
#    if i % 2 == 0:
    _,train_accuacy,y1,y2 = sess.run([train_step,accuracy,prediction,prediction_y],feed_dict={x: batch[0], y_: batch[1]})
    print("step: %s, training accuracy: %s"%(i, train_accuacy))
    print(y1[:5])
    print(y2[:5])

#    else:    
#        train_step.run(feed_dict = {x: batch[0], y_: batch[1]})

# 最后测试准确率
# batch_x_test, batch_y_test = get_batch(100)
# print("test accuracy %g"%(accuracy.eval(feed_dict={x: batch_x_test, y_: batch_y_test})))
    