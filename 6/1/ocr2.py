# coding=utf-8
# 中文OCR学习

import tensorflow as tf
import numpy as np
import os
from utils import readImgFile, img2vec, dropZeroEdges, text2vec, vec2text
import time
import random

curr_dir = os.path.dirname(__file__)

# 图片的高度为20，宽度为1000
image_size = (20,1000)

# 最长100个字节
label_size = 100

# 所有 unicode CJK统一汉字（4E00-9FBB） + ascii的字符加 + blank + ctc blank
# https://zh.wikipedia.org/wiki/Unicode
# https://zh.wikipedia.org/wiki/ASCII
ASCII_CHARS = [chr(c) for c in range(32,126+1)]
ZH_CHARS = [chr(c) for c in range(int('4E00',16),int('9FBB',16)+1)]
ZH_CHARS_PUN = ['。','？','！','，','、','；','：','「','」','『','』','‘','’','“','”',\
                '（','）','〔','〕','【','】','—','…','–','．','《','》','〈','〉']

CHARS = ASCII_CHARS + ZH_CHARS + ZH_CHARS_PUN
CHARS_SIZE = len(CHARS)
#初始化学习速率
# LEARNING_RATE_INITIAL = 1e-3
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 500
# MOMENTUM = 0.9

BATCHES = 64
BATCH_SIZE = 64
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = 10

if os.path.exists(os.path.join(curr_dir, "data", "index.txt")):
    print("Loading data ...")
    train_files = open(os.path.join(curr_dir, "data", "index.txt")).readlines()
else:
    train_files = []

def neural_networks():
    # 训练或学习的样本
    inputs = tf.placeholder(tf.float32, [None, image_size[0]*image_size[1]], name="inputs")
    # 训练的结果
    labels = tf.placeholder(tf.int32, [None, label_size*CHARS_SIZE], name='labels')
    # 卷积层
    filter_sizes = [5, 3, 3]
    filter_nums = [32, 32, 32]
    pool_types = ['avg', 'avg', 'max']
    pool_scale = [2, 2, 2]
    conv_pools = []    
    for i in range(len(filter_sizes)):
        with tf.variable_scope('conv-pool-{}'.format(i)):
            if i == 0:
                input = tf.reshape(inputs, [-1, image_size[0], image_size[1], 1])
            else:
                input = conv_pools[-1]
            filter_shape = [filter_sizes[i], filter_sizes[i],
                            int(input.get_shape()[-1]), filter_nums[i]]
            W = tf.get_variable("filter", filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('bias', [filter_nums[i]], initializer=tf.constant_initializer(0.0))
            W_conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1],  padding='SAME')
            conv = tf.nn.relu(tf.nn.bias_add(W_conv, b))
            if pool_types[i] == 'avg':
                pool = tf.nn.avg_pool(conv, ksize=[1, pool_scale[i], pool_scale[i], 1], strides=[
                                    1, pool_scale[i], pool_scale[i], 1], padding='SAME')
            else:
                pool = tf.nn.max_pool(conv, ksize=[1, pool_scale[i], pool_scale[i], 1], strides=[
                                    1, pool_scale[i], pool_scale[i], 1], padding='SAME')
            conv_pools.append(pool)
    # 全连接层
    hidden_sizes = [256]
    full_connects = []
    for i in range(len(hidden_sizes)):
        with tf.variable_scope('full-connect-{}'.format(i)):
            if i == 0:
                in_size = int(conv_pools[-1].get_shape()[1]) * int(conv_pools[-1].get_shape()[2]) * int(conv_pools[-1].get_shape()[3])
                inputs = tf.reshape(conv_pools[-1], [-1, in_size])
            else:
                inputs = full_connects[-1]
                in_size = hidden_sizes[i - 1]
            W = tf.get_variable("weights", [in_size, hidden_sizes[i]], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("biases", [hidden_sizes[i]], initializer=tf.constant_initializer(0.0))
            full_connect = tf.nn.relu(tf.matmul(inputs, W) + b)
            full_connects.append(full_connect)
    # 由于是多个文本，需要合并输出
    outputs = []
    for i in range(label_size):
        with tf.variable_scope('output-part-{}'.format(i)):
            W = tf.get_variable("weights", [hidden_sizes[-1], CHARS_SIZE], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("biases", [CHARS_SIZE], initializer=tf.constant_initializer(0.0))
            fc_part = tf.matmul(full_connects[-1], W) + b
            outputs.append(fc_part)
    # 输出
    output = tf.concat(outputs, 1)
    # 损失函数
    losses = []
    for i in range(label_size):
        with tf.variable_scope('loss-part-{}'.format(i)):
            outputs_part = tf.slice(output, begin=[0, i * CHARS_SIZE], size=[-1, CHARS_SIZE])
            targets_part = tf.slice(labels, begin=[0, i], size=[-1, 1])
            targets_part = tf.reshape(targets_part, [-1])
            loss_part = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs_part, labels=targets_part)
            reduced_loss_part = tf.reduce_mean(loss_part)
            losses.append(reduced_loss_part)
    loss = tf.reduce_mean(losses, name='loss')
    # 得到最终的验证码
    predictions = []
    for i in range(label_size):
        with tf.variable_scope('predictions-part-{}'.format(i)):
            outputs_part = tf.slice(output, begin=[0, i * CHARS_SIZE], size=[-1, CHARS_SIZE])
            prediction_part = tf.argmax(outputs_part, axis=1)
            predictions.append(prediction_part)
    prediction = tf.stack(predictions, axis=1, name='prediction')

    predictions_y = []
    for i in range(label_size):
        with tf.variable_scope('predictions-y-part-{}'.format(i)):
            outputs_part = tf.slice(labels, begin=[0, i * CHARS_SIZE], size=[-1, CHARS_SIZE])
            prediction_part = tf.argmax(outputs_part, axis=1)
            predictions_y.append(prediction_part)
    prediction_y = tf.stack(predictions_y, axis=1, name='prediction_y')

    # 计算正确率
    correct_prediction = tf.cast(tf.equal(prediction, prediction_y), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction, name='accuracy')  
    return inputs, labels, output, prediction, loss, accuracy

# 生成一个训练batch
def get_next_batch(batch_size=128):
    inputs = np.zeros([batch_size, image_size[1]*image_size[0]])
    labels = np.zeros([batch_size, label_size*CHARS_SIZE])
    batch = random.sample(train_files, batch_size)
    for i, line in enumerate(batch):
        lines = line.split(" ")
        imageFileName = lines[0]+".png"
        text = line[line.index(' '):].strip()
        # 输出图片为反色黑白
        image = readImgFile(os.path.join(curr_dir,"data",imageFileName))   
        image = dropZeroEdges(image)
        inputs[i,:] = img2vec(image,image_size[0],image_size[1])
        labels[i,:] = text2vec(CHARS, text) 
    return inputs, labels

def train():
    global_step = tf.Variable(0, trainable=False)
    
    # 决定还是自定义学习速率比较靠谱                                            
    curr_learning_rate = 1e-3
    learning_rate = tf.placeholder(tf.float32, shape=[])                                            
    inputs, labels, output, prediction, loss, accuracy = neural_networks()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step, name="optimizer")

    init = tf.global_variables_initializer()

    def report_accuracy(_prediction, test_labels, accuracy):
        if len(_prediction) != len(test_labels):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(test_labels):
            label = vec2text(CHARS,number)
            detect_label = "".join([CHARS[s] for s in _prediction[idx]])
            hit = (label == detect_label)
            print(hit, label, "(", len(label), ") <-------> ", detect_label, "(", len(detect_label), ")")
        print("Test Accuracy:", accuracy)

    def do_report():
        test_inputs,test_labels = get_next_batch(TEST_BATCH_SIZE)
        test_feed = {inputs: test_inputs, labels: test_labels}
        _prediction, _accuracy = session.run([prediction, accuracy], test_feed)
        report_accuracy(_prediction, test_labels, _accuracy)
 
    def do_batch():
        train_inputs, train_labels = get_next_batch(BATCH_SIZE)       
        feed = {inputs: train_inputs, labels: train_labels, learning_rate: curr_learning_rate}        
        b_loss, steps, b_learning_rate, _ = session.run([loss, global_step, learning_rate, optimizer], feed)

        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report()
        return b_cost, steps, b_learning_rate

    def restore(sess):
        curr_dir = os.path.dirname(__file__)
        model_dir = os.path.join(curr_dir, "model")
        if not os.path.exists(model_dir): os.mkdir(model_dir)
        saver_prefix = os.path.join(model_dir, "model.ckpt")        
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.Saver(max_to_keep=5)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model ...")
            saver.restore(sess, ckpt.model_checkpoint_path)
        return saver, model_dir, saver_prefix

    with tf.Session() as session:
        session.run(init)
        saver, model_dir, checkpoint_path = restore(session) # tf.train.Saver(tf.global_variables(), max_to_keep=100)
        while True:            
            train_cost = train_ler = 0
            for batch in range(BATCHES):
                start = time.time()
                c, steps, rate = do_batch()
                train_cost += c * BATCH_SIZE
                seconds = round(time.time() - start,2)
                print("step:", steps, "cost:", c, "batch seconds:", seconds, "learning rate:", rate)
                if np.isnan(c):
                    print("Error: cost is nan")
                    return                
            
            # train_cost /= TRAIN_SIZE
                if c < 10 and curr_learning_rate > 1e-4:
                    curr_learning_rate = 1e-4
                if c < 1 and curr_learning_rate > 1e-5:
                    curr_learning_rate = 1e-5
                if c < 0.1 and curr_learning_rate > 1e-6:
                    curr_learning_rate = 1e-6

            # train_inputs, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)
            # val_feed = {inputs: train_inputs,
            #             labels: train_labels,
            #             seq_len: train_seq_len,
            #             input_keep_prob: 1.0  }

            # val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)

            # log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            # print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start, lr))
            saver.save(session, checkpoint_path, global_step=steps)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()