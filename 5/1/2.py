# coding=utf-8

import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
import os
import json

# 使用结巴分词
# pip install jieba
import jieba

curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")
lex_file = os.path.join(curr_dir, "lex.pickle")
dataset_file = os.path.join(curr_dir, "dataset.pickle")
if os.path.exists(lex_file) and os.path.exists(dataset_file):
    lex = pickle.load(open(lex_file))
    dataset = pickle.load(open(dataset_file))
else:
    movies_file = os.path.join(data_dir,"movies.json")
    movies = json.loads(open(movies_file).read())

    # 创建词汇表
    def create_lexicon():    
        lex = []
        for title,_,_ in movies:
            movie_file =os.path.join(data_dir,u"{}.json".format(title))
            movie = json.loads(open(movie_file).read())
            for _,comment in movie:
                lex += jieba.cut(comment)
        word_count = Counter(lex)
        # 直接按词频排序，取 20%~80% 的区间
        words = sorted(word_count, key=word_count.get)
        count = len(words)
        return words[int(count*0.2):int(count*0.8)]

    lex = create_lexicon()
    print(len(lex),"words")
    #lex里保存了文本中出现过的单词。

    #所有的打星
    stars={"allstar10":[1,0,0,0,0],"allstar20":[0,1,0,0,0],"allstar30":[0,0,1,0,0],"allstar40":[0,0,0,1,0],"allstar50":[0,0,0,0,1]}

    # lex:词汇表； comment:评论； star:评论对应的打分 
    def comment_to_vector(lex, comment, star):
        words = jieba.cut(comment)
        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] += 1  #可能有重复
        return [features, stars[star]]

    # 把每条评论转换为向量, 转换原理：
    # 假设 lex 为 ['好', '赞', '太差', '不好看', '垃圾'] 当然实际上要大的多
    # 评论 '我认为这个电影太差了，不好看' 转换为 [0,0,1,1,0], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
    def normalize_dataset(lex):
        dataset = []
        count = len(movies)
        for i, movie in enumerate(movies):
            print(i,"of",count)
            title, _, _ = movie
            movie_file =os.path.join(data_dir,u"{}.json".format(title))
            movie_comments = json.loads(open(movie_file).read())
            for star,comment in movie_comments:
                one = comment_to_vector(lex,comment,star)
                dataset.append(one)   
        print(len(dataset),"records")
        return dataset

    dataset = normalize_dataset(lex)
    random.shuffle(dataset)

    with open(lex_file, 'wb') as f:
        pickle.dump(lex, f)
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f)


# 取样本中的10%做为测试数据
test_size = int(len(dataset) * 0.1)

dataset = np.array(dataset)

train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

# Feed-Forward Neural Network
# 定义每个层有多少'神经元''
n_input_layer = len(lex)  # 输入层

n_layer_1 = 1000    # hide layer
n_layer_2 = 1000    # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 2       # 输出层

# 定义待训练的神经网络
def neural_network(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2 ) # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output

# 每次使用50条数据进行训练
batch_size = 50

X = tf.placeholder('float', [None, len(train_dataset[0][0])]) 
#[None, len(train_x)]代表数据数据的高和宽（矩阵），好处是如果数据不符合宽高，tensorflow会报错，不指定也可以。
Y = tf.placeholder('float')
# 使用数据训练神经网络
def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)  # learning rate 默认 0.001 

    epochs = 13
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        epoch_loss = 0

        i = 0
        random.shuffle(train_dataset)
        train_x = dataset[:, 0]
        train_y = dataset[:, 1]
        for epoch in range(epochs):
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]

                _, c = session.run([optimizer, cost_func], feed_dict={X:list(batch_x),Y:list(batch_y)})
                epoch_loss += c
                i += batch_size

            print(epoch, ' : ', epoch_loss)

        text_x = test_dataset[: ,0]
        text_y = test_dataset[:, 1]
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('准确率: ', accuracy.eval({X:list(text_x) , Y:list(text_y)}))

train_neural_network(X,Y)