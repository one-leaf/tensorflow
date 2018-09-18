#!/usr/bin/python3
'''
5 训练 word2vec 模型，输出 embedding.npy
模型参考：
https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

skip-gram 模型使用 tf.nn.nce_loss
cbow 模型使用 tf.nn.sampled_softmax_loss 
'''

import os
import collections
import math
import random
import numpy as np
from six.moves import xrange
import tensorflow as tf
import time


curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")
words_filename = os.path.join(data_dir,"words.txt")   
sentence_filename = os.path.join(data_dir,"sentence.txt")   
model_dir = os.path.join(curr_dir, "model", "word2vec")
if not os.path.exists(model_dir): os.mkdir(model_dir)

# 读取words
def read_words():
    word_list = []
    with open(words_filename,"r", encoding='UTF-8') as f:
        for line in f:
            if line!="":
                word_list.append(line.strip())
    return word_list

#step 1:读取文件中的内容组成一个列表
words = read_words()
vocabulary_size = len(words)
print('words size', vocabulary_size)

# Step 2: 根据wold建立索引, 0为UNK 
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

# Step 3: 产生训练样本
# batch_size  每一批数据的大小
def generate_batch(batch_size=128):
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 3

    batch_index = 0
    with open(sentence_filename,"r", encoding='UTF-8') as f:
        lines = f.readlines()

    random.shuffle(lines)
    for line in lines:
        # 准备数据
        words = line.split(" ")
        data = []
        for word in words:
            word = word.strip()
            if word in dictionary:  # 如果不存在，编号就是0，表示未知
                data.append(dictionary[word])
            else:
                data.append(0)

        if len(data) < span : continue # 如果一行的单词个数小于3个，忽略           
        for i in range(len(data)-span+1):
            target = 1+i   # 中间的目标 label [1]
            targets_to_avoid = [i, 2+i]  # [1]

            for j in targets_to_avoid:  #单词： a b c 
                batch[batch_index] = data[j] #取边缘值 a 和 c
                labels[batch_index, 0] = data[target]   # 取中间词 b
                batch_index += 1

                if batch_index == batch_size :
                    yield batch, labels
                    batch_index = 0

batch, labels = next(generate_batch(batch_size=8))
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],'->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: 构建和训练 a skip-gram model.
batch_size = 256
embedding_size = 256    
num_sampled = batch_size//2    # 反例的个数 是 batch_size // 2

#验证集,这里验证下数字是否会归类到一起
valid_word = ['0','1','2','.','型号','材质','用途','其他','品牌','厂商']
valid_size = len(valid_word)  
valid_examples =[dictionary[li] for li in valid_word]

graph = tf.Graph()
with graph.as_default():
    # 输入数据.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 不支持GPU,放开也没有意义
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]),dtype=tf.float32)

    loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,biases=nce_biases, inputs=embed, labels=train_labels,
                 num_sampled=num_sampled, num_classes=vocabulary_size))

    lr = tf.Variable(1.0, trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

# Step 5: 开始训练.
num_epochs = 100
with tf.Session(graph=graph) as session:
    init.run()
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
                f.write('model_checkpoint_path: "word2vec.ckpt-%s"\n'%step)
                f.write('all_model_checkpoint_paths: "word2vec.ckpt-%s"\n'%step)
            raise Exception("can't resore model restart")

    g_batch = generate_batch(batch_size)
    average_loss = 0
    start = time.time() 
    for epoch in xrange(num_epochs):

        for batch_inputs, batch_labels in generate_batch(batch_size):
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, loss_val, lr_val = session.run([optimizer, loss, lr], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print(time.ctime(), "step:", epoch, "/", step, "loss:", average_loss, "time:", time.time() - start, "lr:", lr_val)
                average_loss = 0
                start = time.time() 

            # 校验
            if step % 100000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 20  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[:top_k]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
            
                # save model
                saver.save(session, os.path.join(model_dir, "word2vec.ckpt"), global_step=step) 

                # 每一轮训练，学习速率降低一次， 按50万下降
            #    session.run(tf.assign(lr, 0.1 * pow(0.99, step/1000000)))

            step += 1
        final_embeddings = normalized_embeddings.eval()
        # 保存模型
        np.save(os.path.join(data_dir,"embedding.npy"), final_embeddings)
  
