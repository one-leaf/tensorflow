# coding=utf-8
# 卷积多层多项验证码识别

from generate_captcha import gen_captcha_text_and_image as captcha
import numpy as np
from utils import img2gray, img2vec, text2vec, vec2text
import tensorflow as tf
import os
import datetime

image_h = 80                    # 图片高
image_w = 200                   # 图片宽
image_size = image_h * image_w  # 图片大小
char_set = "0123456789"         # 验证码组成
char_size = len(char_set)       # 验证码组成种类长度
captcha_size = 4                # 验证码长度
batch_size = 100                # 每一批学习的验证码个数

# 是否开启调试 到程序目录执行 tensorboard --logdir=summaries ，访问 http://127.0.0.1:6006
DEBUG = True

# 批量验证码数据
def get_batch(batch_size=128):
    batch_x = np.zeros([batch_size, image_size])
    batch_y = np.zeros([batch_size, captcha_size])
    for i in range(batch_size):
        text, image = captcha(
            char_set=char_set, captcha_size=captcha_size, width=image_w, height=image_h)
        batch_x[i, :] = img2vec(img2gray(image))
        batch_y[i, :] = list(text)  # 注意 这里的 lable 不能 one hot
    return batch_x, batch_y

# 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，
# 第一维是 batch_size 每次训练的样本数， 第2、3维对应图片的高和宽，
# 最后一维对应颜色通道的数目，这里是黑白，所以为 1 ，如果图片为 RGB 则为3 。
x = tf.placeholder(tf.float32, [None, image_size])
x_ = tf.reshape(x, [batch_size, image_h, image_w, 1])
y_ = tf.placeholder(tf.int32, [batch_size, captcha_size])

# 输出原始图片
if DEBUG:
    img = tf.expand_dims(x_[-1], 0)  # (h,w,c) => (1,h,w,c)      
    tf.summary.image('x', tensor=img,  max_outputs=1)

# 卷积层
filter_sizes = [5, 5, 3, 3]
filter_nums = [32, 32, 32, 32]
pool_types = ['avg', 'max', 'max', 'max']
pool_scale = [2, 2, 2, 2]
conv_pools = []
for i in range(len(filter_sizes)):
    with tf.variable_scope('conv-pool-{}'.format(i)):
        if i == 0:
            input = x_
        else:
            input = conv_pools[-1]
        filter_shape = [filter_sizes[i], filter_sizes[i],
                        int(input.get_shape()[-1]), filter_nums[i]]
        # tf.contrib.layers.xavier_initializer_conv2d 按照 Xavier
        # 方式初始化，好处是在所有层上保持大致相同的梯度，机器学习专用
        W = tf.get_variable(
            "filter", filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable('bias', [filter_nums[i]],
                            initializer=tf.constant_initializer(0.0))
        W_conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1],  padding='SAME')
        conv = tf.nn.relu(tf.nn.bias_add(W_conv, b))
        if pool_types[i] == 'avg':
            pool = tf.nn.avg_pool(conv, ksize=[1, pool_scale[i], pool_scale[i], 1], strides=[
                                  1, pool_scale[i], pool_scale[i], 1], padding='SAME')
        else:
            pool = tf.nn.max_pool(conv, ksize=[1, pool_scale[i], pool_scale[i], 1], strides=[
                                  1, pool_scale[i], pool_scale[i], 1], padding='SAME')
        conv_pools.append(pool)

        if DEBUG:
            filter_map = conv[-1]       # shape: [h, w, filter_nums]
            filter_map = tf.transpose(filter_map, perm=[2, 0, 1])   # shape: [filter_nums, h, w]
            filter_map = tf.reshape(filter_map, (filter_nums[i], int(filter_map.get_shape()[
                                    1]), int(filter_map.get_shape()[2]), 1))  # [filter_nums, h, w, c:1]
            tf.summary.image('conv', tensor=filter_map,
                             max_outputs=filter_nums[i])
            # 没有必要看 pool 层，直接看 conv 层足够了
            # filter_map = pool[-1]
            # filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
            # filter_map = tf.reshape(filter_map, (filter_nums[i], int( filter_map.get_shape()[1]), int(filter_map.get_shape()[2]), 1))
            # tf.summary.image('conv-pool', tensor=filter_map,  max_outputs=filter_nums[i])
            # 输出 W，b 到 tensorboard 实际训练时，关闭这个
            tf.summary.histogram('W'.format(i), W)
            tf.summary.histogram('b'.format(i), b)

# 全连接层
hidden_sizes = [256]
full_connects = []
for i in range(len(hidden_sizes)):
    with tf.variable_scope('full-connect-{}'.format(i)):
        if i == 0:
            batch_size = int(x_.get_shape()[0])
            inputs = tf.reshape(conv_pools[-1], [batch_size, -1])
            in_size = int(inputs.get_shape()[-1])
        else:
            inputs = full_connects[-1]
            in_size = hidden_sizes[i - 1]
        # 常用初始化函数
        # tf.constant_initializer(value) 初始化一切所提供的值
        # tf.random_uniform_initializer(a, b)从a到b均匀初始化
        # tf.random_normal_initializer(mean, stddev) 用所给平均值和标准差初始化均匀分布
        # tf.contrib.layers.xavier_initializer 按照 Xavier
        # 方式初始化，好处是在所有层上保持大致相同的梯度，机器学习专用
        W = tf.get_variable("weights", [in_size, hidden_sizes[
                            i]], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(
            "biases", [hidden_sizes[i]], initializer=tf.constant_initializer(0.0))
        # 常用激活函数
        # sigmoid   能够把输入的连续实值“压缩”到0和1之间，容易过饱和和信息丢失，很少使用
        # tanh      是 sigmoid 的变形，比 sigmoid 好
        # relu      收敛快，但不能将 learning_rate 设置太大，会导致梯度直接为0，推荐
        # relu6     比 relu 的收敛更快
        full_connect = tf.nn.relu(tf.matmul(inputs, W) + b)
        # full_connect = tf.nn.sigmoid(tf.matmul(inputs, W) + b)
        # full_connect = tf.nn.tanh(tf.matmul(inputs, W) + b)
        full_connects.append(full_connect)

        if DEBUG:
            tf.summary.histogram('W', W)
            tf.summary.histogram('b', b)

# 由于是多位验证码，继续添加每一位验证码的输出
outputs = []
for i in range(captcha_size):
    with tf.variable_scope('output-part-{}'.format(i)):
        W = tf.get_variable("weights", [
                            hidden_sizes[-1], char_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("biases", [char_size],
                            initializer=tf.constant_initializer(0.0))
        fc_part = tf.matmul(full_connects[-1], W) + b
        outputs.append(fc_part)

        if DEBUG:
            tf.summary.histogram('W', W)
            tf.summary.histogram('b', b)

# 最终输出
output = tf.concat(outputs, 1)

# 抛弃函数
losses = []
for i in range(captcha_size):
    with tf.variable_scope('loss-part-{}'.format(i)):
        outputs_part = tf.slice(
            output, begin=[0, i * char_size], size=[-1, char_size])
        targets_part = tf.slice(y_, begin=[0, i], size=[-1, 1])
        targets_part = tf.reshape(targets_part, [-1])
        # 常用损失函数
        # sigmoid_cross_entropy_with_logits 二分类问题,不支持多分类
        # softmax_cross_entropy_with_logits 只适合单目标的二分类或者多分类问题，多分类问题需要做成 onehot
        # sparse_softmax_cross_entropy_with_logits 同上，只是分类目标不需要 onehot
        # weighted_cross_entropy_with_logits 基于 sigmoid_cross_entropy_with_logits ，
        #   多支持一个pos_weight参数，目的是可以增加或者减小正样本在算Cross Entropy时的Loss
        loss_part = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=outputs_part, labels=targets_part)
        # loss_part = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs_part, labels=targets_part)
        # loss_part = tf.nn.softmax_cross_entropy_with_logits(logits=outputs_part, labels=targets_part) #这里不适用，需要 onehot 化才可以
        # loss_part = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs_part, labels=targets_part)
        # loss_part = tf.nn.weighted_cross_entropy_with_logits(logits=outputs_part, labels=targets_part)
        reduced_loss_part = tf.reduce_mean(loss_part)
        losses.append(reduced_loss_part)
loss = tf.reduce_mean(losses)

# 得到最终的验证码
predictions = []
for i in range(captcha_size):
    with tf.variable_scope('predictions-part-{}'.format(i)):
        outputs_part = tf.slice(
            output, begin=[0, i * char_size], size=[-1, char_size])
        prediction_part = tf.argmax(outputs_part, axis=1)
        prediction_part = tf.cast(prediction_part, tf.int32)
        predictions.append(prediction_part)
prediction = tf.stack(predictions, axis=1)

# 计算正确率
correct_prediction = tf.cast(tf.equal(prediction, y_), tf.float32)
correct_prediction = tf.reduce_mean(correct_prediction, axis=1)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(correct_prediction, 1.0), tf.float32))

# 定义学习速率和优化方法
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    1e-3, global_step, 2000, 0.96, staircase=True)
# 学习速率调整函数
# MomentumOptimizer  动量算法，开始会震荡，后面随着动量的增加，震荡会减少，快速下降
# GradientDescentOptimizer 梯度下降算法，速度较慢，稳定
# AdagradOptimizer   利用梯度信息调整学习速率，适合比较稀疏的数据，比较稳定
# RMSPropOptimizer   解决AdaGrad中学习速率趋向0的问题
# AdadeltaOptimizer  也可以解决AdaGrad的问题，和RMSPropOptimizer类似，但更高级，不需要初始的学习速率
# AdamOptimizer      利用了AdaGrad和RMSProp在稀疏数据上的优点，可以进行自适应调整 推荐
# FtrlOptimizer      在线最优化算法，计算量小，适合高纬度的稀疏数据
# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# optimizer = tf.train.AdagradOptimizer(learning_rate)
# optimizer = tf.train.RMSPropOptimizer(learning_rate)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.FtrlOptimizer(learning_rate)
train_step = optimizer.minimize(loss, global_step=global_step)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 定义中断后的恢复和继续训练
# max_to_keep 最多保留的检查点数量，下面是 5 个
# keep_checkpoint_every_n_hours 每隔多少小时至少保留一个检查点，下面是每隔 1 小时
saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
out_dir = os.path.dirname(__file__)
checkpoints_dir = os.path.join(out_dir, "checkpoints")
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)
checkpoint_prefix = os.path.join(checkpoints_dir, "model")
# 检查到如果存在检查点，就装载继续运行
ckpt = tf.train.get_checkpoint_state(checkpoints_dir)
if ckpt and ckpt.model_checkpoint_path:
    print("restore checkpoint and continue train.")
    saver.restore(sess, ckpt.model_checkpoint_path)

# 定义图表输出
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
train_summary_op = tf.summary.merge_all()
train_summary_dir = os.path.join(out_dir, "summaries")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# 定义运行协调
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# 运行
try:
    while not coord.should_stop():
        batch = get_batch(batch_size)
        _, _train_summaries, _loss, _preds, _acc, _step = sess.run(
            [train_step, train_summary_op, loss,
                predictions, accuracy, global_step],
            feed_dict={x: batch[0], y_: batch[1]})
        time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
        print(time_str, _step, _acc, _loss)
        train_summary_writer.add_summary(_train_summaries, _step)
        if _step % 10 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=_step)
            print("save train model to", path)
finally:
    coord.request_stop()
coord.join(threads)

# 结束学习，关闭日志和会话
train_summary_writer.close()
sess.close()
