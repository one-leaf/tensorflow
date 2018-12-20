# word2vec

import os
import re
import jieba
import tensorflow as tf
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 词汇表长度
vocabulary_size = 800

# 词向量维度
embedding_size = 16

curr_path = os.path.dirname(os.path.realpath(__file__))
txt_file = os.path.join(curr_path,'../12-应用.md')
sentences = open(txt_file, encoding="UTF-8").readlines()

def split(sentence):
    s = sentence.strip()
    s = s.lower()
    s = re.sub(r'[!@#$%^&*(){}:/";\\_\+\=\-\'<>|\?,.\[\]，。；（）、：]', ' ', s)
    _words = jieba.lcut(s)
    if ' ' in _words:
        _words.remove(' ')
    return _words

def getWords(vocabulary_size):
    words={}
    
    dictionary = {}
    for s in sentences:
        _words = split(s)
        for word in _words:
            word = word.strip()
            if word == '': continue
            if word in dictionary:
                dictionary[word]+=1
            else:
                dictionary[word]=1
    print("dictionary length:",len(dictionary))

    sorted_words = sorted(dictionary, key=dictionary.get, reverse=True)[:vocabulary_size-1]

    for i, w in enumerate(sorted_words):
        words[w]=i
    words['UNK']=len(words)
    return words

'''
    产生样本 
    skip_window ：目标词前后关联范围大小   [ skip_window skip_window target skip_window skip_window]
    num_skips ： 从一个总span中取多少个样本
'''
def generate_batch(words, batch_size, skip_window = 1, num_skips = 2):    
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    batch_index = 0
   
    # 总长 [ skip_window target skip_window ] 
    span = 2 * skip_window + 1   

    assert num_skips <= 2 * skip_window

    for s in sentences:

        data=[] # 词列表
        # 获得每一句的词汇表
        _words = split(s)
        for word in _words:
            word = word.strip()
            if word == '': continue
            if word in words:
                data.append(words[word])
            else:
                data.append(words['UNK'])
        
        # 如果词汇表太短了，小于总长，放弃
        if len(data)<span: continue

        # 向前滑动抓数据
        for i in range(len(data)-span+1):
            buffer = data[i:i+span]
            # 上下文词汇表
            context_words = [w for w in range(span) if w != skip_window]
            # 需要采样的上下文词汇
            words_to_use = random.sample(context_words, num_skips)
            # 按词汇表进行采样
            for j, context_word in enumerate(words_to_use):
                # 中心
                batch[batch_index] = buffer[skip_window]
                labels[batch_index, 0] = buffer[context_word]
                batch_index += 1               

                if batch_index == batch_size :
                    yield batch, labels
                    batch_index = 0            


class network():
    def __init__(self, vocabulary_size, embedding_size, num_sampled):
        self.x = tf.placeholder(tf.int32, [None], name='x')
        self.y = tf.placeholder(tf.int32, [None, 1], name='y')

        self.embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        self.nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        embed = tf.nn.embedding_lookup(self.embeddings, self.x)
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,
                                            biases=self.nce_biases,
                                            labels=self.y,
                                            inputs=embed,
                                            num_sampled=num_sampled,
                                            num_classes=vocabulary_size))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(self.loss)
        
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings, self.x)
        self.similarity = tf.matmul(valid_embeddings, self.normalized_embeddings, transpose_b=True)


def main():
    words = getWords(vocabulary_size)
    print(words)

    reversed_words = dict(zip(words.values(), words.keys()))
    print(reversed_words)

    # 每批大小
    batch_size = 32
    # 反例大小
    num_sampled = batch_size//2
    # 检查大小
    valid_size = 1

    net = network(vocabulary_size, embedding_size, num_sampled)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        avg_loss = 0
        for epoch in range(200):
            for batch, labels in generate_batch(words, batch_size, 2, 3):
                loss,_= sess.run([net.loss, net.optimizer], feed_dict={net.x: batch, net.y: labels})

                if avg_loss==0:
                    avg_loss=loss
                else:
                    avg_loss = avg_loss*0.999+ loss*0.001

            # 随机抓前100个词中的 valid_size 个测试
            batch = np.random.choice(100, valid_size, replace=False)
            similarity = sess.run(net.similarity, feed_dict={net.x: batch})
            for i in range(valid_size):
                nearest = (-similarity[i, :]).argsort()[:10]
                nearest_words = [reversed_words[idx] for idx in nearest]
                print(epoch,"loss：", avg_loss, "valid：", " ".join(nearest_words))

        # 最终归一化输出词向量矩阵
        final_embeddings = net.normalized_embeddings.eval()

        # 计算两个词的相似度
        similarity_words = [("神经网络","单元"),("神经网络","循环"),("单元","循环")]
        for word1,word2 in similarity_words:
            vec1 = final_embeddings[words[word1]]
            vec2 = final_embeddings[words[word2]]
            print(word1, word2, '欧式距离:', np.linalg.norm(vec1 - vec2))

        # 可视化词语之间的关系
        plot_only = 200
        # 降维
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
        labels = [reversed_words[i] for i in range(plot_only)]
        plt.rcParams['font.sans-serif']=['SimHei','SimSun'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        plt.figure()  
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')
        plt.show()

if __name__ == "__main__":
    main()