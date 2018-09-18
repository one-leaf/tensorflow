#!/usr/bin/python3
'''
根据句子预测分类
'''

import os
import collections
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim 
from tensorflow.contrib.slim import nets 
import time
import json
import copy

curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")
words_filename = os.path.join(data_dir,"words.txt")   
embed_filename = os.path.join(data_dir,"embedding.npy")   
dataset_filename = os.path.join(data_dir,"dataset.txt")   
model_dir = os.path.join(curr_dir, "model", "declare")
if not os.path.exists(model_dir): os.makedirs(model_dir)

# 读取words
def load_words():
    word_list = []
    with open(words_filename,"r", encoding='UTF-8') as f:
        for line in f:
            if line!="":
                word_list.append(line.strip())
    return word_list

words = load_words()
vocabulary_size = len(words)
print('words size', vocabulary_size)

# 根据wold建立索引, 0为UNK 
def build_dataset(words):
    dictionary = dict()
    for i, word in enumerate(words):
        dictionary[word]=i
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(words)
print('dictionary size', len(dictionary))
print("0~10:",[reverse_dictionary[i] for i in range(10)])
del words  #删除words节省内存

# 加载word向量
def load_embed():
    return np.load(embed_filename)
embed = load_embed()
print('embed size', embed.shape)

# 由word转word向量
def getWord2Vec(word):
    if word in dictionary:
        return embed[dictionary[word]]
    else:
        return embed[0]

# 定义神经网络
# cnn resnet v2
def addResLayer(inputs):
    layer = slim.batch_norm(inputs, activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.conv2d(layer, 64, [3,3], activation_fn=None)
    layer = slim.batch_norm(layer, activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.conv2d(layer, 64, [3,3], activation_fn=None)
    outputs = inputs + layer
    return outputs  

# RNN 求最终结果值
def LSTM(inputs, seq_len, reuse=False, lstm_size=128):
    layer = inputs
    with tf.variable_scope("rnn", reuse=reuse):
        cell_fw = tf.contrib.rnn.GRUCell(lstm_size)
        cell_bw = tf.contrib.rnn.GRUCell(lstm_size)
        _, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, layer, sequence_length=seq_len,  dtype=tf.float32)
    layer = tf.concat(output_states, -1)
    return layer  

# 需要识别的最长单词个数
MAX_HSMODEL_LEN = 20
def networks():
    # 输入为2个序列，其中第一个为待分类文本，后面为20个目标分类
    # 目标字符串
    inputs_declare = tf.placeholder(tf.float32, [None, None, 256], name="inputs_declare")
    inputs_declare_seq_len = tf.placeholder(tf.int32, [None], name="inputs_declare_seq_len")

    inputs_declare_other = tf.placeholder(tf.float32, [None, None, 256], name="inputs_declare_other")
    inputs_declare_other_seq_len = tf.placeholder(tf.int32, [None], name="inputs_declare_other_seq_len")

    # 规范申报类型
    inputs_hsmodel = tf.placeholder(tf.float32, [None, MAX_HSMODEL_LEN, None, 256], name="inputs_hsmodel")
    inputs_hsmodel_seq_len = tf.placeholder(tf.int32, [None, MAX_HSMODEL_LEN], name="inputs_hsmodel_seq_len")


    # 输出为20个分类的softmax
    labels = tf.placeholder(tf.float32, [None, MAX_HSMODEL_LEN], name="labels")

    inputs_declare_rnn = LSTM(inputs_declare, lstm_size= 256/2, seq_len=inputs_declare_seq_len)   #[bacth, 256]
    print("inputs_declare_rnn", inputs_declare_rnn.shape)

    inputs_declare_other_rnn =  LSTM(inputs_declare_other, lstm_size= 256/2, seq_len=inputs_declare_other_seq_len, reuse=True) #[bacth, 256]
    print("inputs_declare_other_rnn", inputs_declare_other_rnn.shape)

    inputs_hsmodel_layers = tf.unstack(inputs_hsmodel, axis=1)          #[batch,256] of MAX_HSMODEL_LEN
    print("inputs_hsmodel_layers", len(inputs_hsmodel_layers), "of", inputs_hsmodel_layers[0].shape,)
    inputs_hsmodel_seq_len_layers = tf.unstack(inputs_hsmodel_seq_len, axis=1)  #[batch] of MAX_HSMODEL_LEN
    print("inputs_hsmodel_seq_len_layers", len(inputs_hsmodel_seq_len_layers), "of", inputs_hsmodel_seq_len_layers[0].shape,)

    inputs_hsmodel_rnn_layers=[]
    inputs_rnn_mse_list=[]
    for i in range(MAX_HSMODEL_LEN):
        inputs_hsmodel_rnn =  LSTM(inputs_hsmodel_layers[i], lstm_size= 256/2, seq_len=inputs_hsmodel_seq_len_layers[i], reuse=True) #[bacth, 256]
        inputs_hsmodel_rnn_concat = tf.concat([inputs_declare_rnn, inputs_declare_other_rnn, inputs_hsmodel_rnn], -1) #[batch, 256*3]
        inputs_hsmodel_rnn_concat = tf.reshape(inputs_hsmodel_rnn_concat,(-1, 256, 3))
        inputs_hsmodel_rnn_layers.append(inputs_hsmodel_rnn_concat)

        # 求向量的均方距离
        inputs_rnn_mse = tf.reduce_mean(tf.square(inputs_declare_rnn-inputs_hsmodel_rnn), axis=1)
        inputs_rnn_mse_list.append(inputs_rnn_mse)

    # 求向量距离差的权重
    inputs_rnn_mse_list =  tf.stack(inputs_rnn_mse_list, -1)     #[batch, MAX_HSMODEL_LEN]

    layer = tf.stack(inputs_hsmodel_rnn_layers, 1) # [batch, MAX_HSMODEL_LEN, 256, 3]   
    print("concat",layer.shape)

    layer = tf.reshape(layer,(-1, MAX_HSMODEL_LEN*4, 256//4, 3))  # [batch, 80, 64, 3]

    print("cnn_before",layer.shape)

    # 这里只能用 avg_pool2d ，不能用 max_pool2d ，会丢失信息，后期很容易出现loss inf
    print("cnn_before",layer.shape)
    layer = slim.conv2d(layer, 64, [3, 3], activation_fn=None)
    for i in range(5):
        with tf.variable_scope("cnn-%s"%i):
            layer = addResLayer(layer)
            layer = slim.avg_pool2d(layer, [3, 3], [2,2], padding = "SAME")      # [batch, 3, 3, 64] 
    print("cnn_res_end",layer.shape)

    layer = tf.layers.flatten(layer)            # [batch, 576]
    print("flatten",layer.shape)

    layer = slim.fully_connected(layer, 1024, activation_fn=tf.nn.relu) #[batch, 1024]
    print("fullconnect",layer.shape)

    logits = slim.fully_connected(layer, MAX_HSMODEL_LEN, activation_fn=None) #[batch, 20] 
    print("logits",logits.shape)

    # 这里假设如果分类正确，那么两段语义上也应该是相似的
    # 这个的好处是将语义直接合并到结果
    logits = logits - inputs_rnn_mse_list
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient(labels)))
    prediction = slim.layers.softmax(logits)

    # 引入 L2 正则
    tv = tf.trainable_variables()
    regularization_cost = 0.001 * tf.reduce_mean([ tf.nn.l2_loss(v) for v in tv ]) 
    cost = cost + regularization_cost

    optimizer = tf.train.AdamOptimizer(0.000001).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return inputs_declare, inputs_declare_seq_len, \
           inputs_declare_other, inputs_declare_other_seq_len, \
           inputs_hsmodel, inputs_hsmodel_seq_len, \
           labels, prediction, optimizer, cost, accuracy

# 读取一批数据
def generate_batch(batch_size=128):
    batch_index = 0
    inputs_declare = [[] for i in range(batch_size)]
    inputs_declare_other = [[] for i in range(batch_size)]
    inputs_hsmodel = [[] for i in range(batch_size)]
    labels = np.zeros(shape=(batch_size, MAX_HSMODEL_LEN), dtype=np.float32)
    inputs_declare_seq_len = np.zeros(batch_size, dtype=np.int32)
    inputs_declare_other_seq_len = np.zeros(batch_size, dtype=np.int32)
    inputs_hsmodel_seq_len = np.zeros((batch_size, MAX_HSMODEL_LEN), dtype=np.int32)

    with open(dataset_filename, "r", encoding='UTF-8') as f:
        lines = f.readlines()

    # 打乱数据
    random.shuffle(lines)
    for line in lines:
        # 准备数据
        data = json.loads(line)

        if random.random()>0.2:
            declare_list = data[1]  #待预测数据 [[],[]...]...[[],[]...] 
            hsmodel_list = data[0]  #规范申报栏位，和待预测数据1：1对应
        else:
            declare_list = data[0]
            hsmodel_list = data[1]

        # 如果只有一项数据，不要训练了
        # if len(hsmodel_list)==1: continue
        for i, declare in enumerate(declare_list):    
            for w in declare:
                inputs_declare[batch_index].append(getWord2Vec(w))
            inputs_declare_seq_len[batch_index] = len(inputs_declare[batch_index])

            # 采集相关参考数据，做出最佳判断
            declare_list_other_indexs = list(range(len(declare_list)))
            declare_list_other_indexs.remove(i)
            if len(declare_list_other_indexs)>0:
                declare_list_other_indexs = random.sample(declare_list_other_indexs, random.randint(0, len(declare_list_other_indexs)))
            for index in declare_list_other_indexs:
                for w in declare_list[index]:
                    inputs_declare_other[batch_index].append(getWord2Vec(w))
            inputs_declare_other_seq_len[batch_index] = len(inputs_declare_other[batch_index])

            # 当前预测值对应的规范申报内容
            hsmodel_list_indexs = list(range(len(hsmodel_list)))    #[0,1,2,3,...]
            if len(hsmodel_list)>2:
                hsmodel_list_indexs.remove(i)
                hsmodel_list_indexs = random.sample(hsmodel_list_indexs, random.randint(1,len(hsmodel_list_indexs)))
                hsmodel_list_indexs.append(i)
            random.shuffle(hsmodel_list_indexs)
            for j, index in enumerate(hsmodel_list_indexs):
                inputs_hsmodel[batch_index].append([])
                for w in hsmodel_list[index]:
                    inputs_hsmodel[batch_index][-1].append(getWord2Vec(w))
                inputs_hsmodel_seq_len[batch_index][j] = len(inputs_hsmodel[batch_index][-1])

            labels_vec = np.zeros((MAX_HSMODEL_LEN), dtype=np.float32)
            # 找到当前乱序后应该对应的标记
            labels_vec[hsmodel_list_indexs.index(i)] = 1.0
            labels[batch_index] = labels_vec

            batch_index += 1

            if batch_index == batch_size :
                if random.random()>0.99:
                    print("inputs_declare", declare, ":", "/".join(hsmodel_list[i]))
                    print("declare_list_other","  ".join(["/".join(declare_list[j]) for j in declare_list_other_indexs]))
                    print("hsmodel_list","  ".join(["/".join(hsmodel_list[j]) for j in hsmodel_list_indexs]))

                inputs_declare_vec = np.zeros(shape=(batch_size, max(inputs_declare_seq_len),  256), dtype=np.float32)
                inputs_declare_other_vec = np.zeros(shape=(batch_size, max(inputs_declare_other_seq_len),  256), dtype=np.float32)
                inputs_hsmodel_vec = np.zeros(shape=(batch_size, MAX_HSMODEL_LEN, max(inputs_hsmodel_seq_len.flatten()), 256), dtype=np.float32)
                for j in range(batch_size):
                    for k,v in enumerate(inputs_declare[j]): 
                        inputs_declare_vec[j][k] = v
                    for k,v in enumerate(inputs_declare_other[j]): 
                        inputs_declare_other_vec[j][k] = v
                    for k, inputs_hsmodel_item in enumerate(inputs_hsmodel[j]): 
                        for l, v in enumerate(inputs_hsmodel_item):
                            inputs_hsmodel_vec[j][k][l] = v

                yield inputs_declare_vec, inputs_declare_seq_len, \
                    inputs_declare_other_vec, inputs_declare_other_seq_len, \
                    inputs_hsmodel_vec, inputs_hsmodel_seq_len, labels
                batch_index = 0
                inputs_declare = [[] for _ in range(batch_size)]
                inputs_declare_other = [[] for _ in range(batch_size)]
                inputs_hsmodel = [[] for _ in range(batch_size)]
                labels = np.zeros(shape=(batch_size, MAX_HSMODEL_LEN), dtype=np.float32)
                inputs_declare_seq_len = np.zeros(batch_size, dtype=np.int32)
                inputs_declare_other_seq_len = np.zeros(batch_size, dtype=np.int32)
                inputs_hsmodel_seq_len = np.zeros((batch_size, MAX_HSMODEL_LEN), dtype=np.int32)

# 训练
def train():
    inputs_declare, inputs_declare_seq_len, \
        inputs_declare_other, inputs_declare_other_seq_len, \
        inputs_hsmodel, inputs_hsmodel_seq_len, \
        labels, prediction, optimizer, cost, accuracy = networks()

    session = tf.Session() 
    session.run(tf.global_variables_initializer())
    print("Initialized")

    step = 0
    saver = tf.train.Saver(max_to_keep=5)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restore Model", ckpt.model_checkpoint_path)
        stem = os.path.basename(ckpt.model_checkpoint_path)
        step = int(stem.split('-')[-1])
        try:
            saver.restore(session, ckpt.model_checkpoint_path)    
        except:
            filenames  = os.listdir(model_dir)
            pre_step = 0 
            for filename in filenames:
                if not filename.endswith("index"): continue
                filename = filename.split('.')[1]
                filename = filename.split('-')[1]
                fileno = int(filename)
                if fileno < step and fileno > pre_step:
                    pre_step = fileno
            step = pre_step
            with open(os.path.join(model_dir,"checkpoint"),'w') as f:
                f.write('model_checkpoint_path: "declare.ckpt-%s"\n'%step)
                f.write('all_model_checkpoint_paths: "declare.ckpt-%s"\n'%step)
            raise Exception("can't resore model restart")

    batch_size = 128
    num_epochs = 100
    g_batch = generate_batch(batch_size)
    start = time.time() 
    avg_loss = 0
    avg_acc = 0
    for epoch in range(num_epochs):

        for batch in generate_batch(batch_size):

            batch_inputs_declare, batch_inputs_declare_seq_len, \
                    batch_inputs_declare_other, batch_inputs_declare_other_seq_len, \
                    batch_inputs_hsmodel, batch_inputs_hsmodel_seq_len, batch_labels = batch

            feed_dict = {inputs_declare: batch_inputs_declare, 
                        inputs_declare_seq_len: batch_inputs_declare_seq_len,
                        inputs_declare_other: batch_inputs_declare_other, 
                        inputs_declare_other_seq_len: batch_inputs_declare_other_seq_len,
                        inputs_hsmodel: batch_inputs_hsmodel, 
                        inputs_hsmodel_seq_len: batch_inputs_hsmodel_seq_len,
                        labels: batch_labels}

            _, loss_val, acc_val = session.run([optimizer, cost, accuracy], feed_dict=feed_dict)

            if avg_loss==0:
                avg_loss = loss_val
                avg_acc = acc_val
            else:
                alpha = 0.0001
                avg_loss = loss_val*alpha + (1.0-alpha)*avg_loss
                avg_acc =   acc_val*alpha + (1.0-alpha)*avg_acc

            if step % 10 == 0:
                print(time.ctime(), "step:", epoch, "/", step, "loss:", loss_val, "/" , avg_loss, "time:", time.time() - start, "acc:", acc_val, "/", avg_acc)
            start = time.time() 

            if np.isnan(loss_val) or np.isinf(loss_val) :
                print("Error: loss is nan or inf，loss:",loss_val)
                return
            # 校验
            if step % 1000 == 0:               
                # save model
                saver.save(session, os.path.join(model_dir, "declare.ckpt"), global_step=step) 
                # print(batch_inputs_declare[0])
                # print(batch_inputs_declare_other[0])
                # print(batch_inputs_hsmodel[0])
                print(batch_labels[-1], 'declare:', batch_inputs_declare_seq_len[-1], \
                            'other:', batch_inputs_declare_other_seq_len[-1],\
                            'hsmodel:', batch_inputs_hsmodel_seq_len[-1])
            step += 1
 
if __name__ == '__main__':
    train()