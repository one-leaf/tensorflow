# coding=utf-8
# 本例中采用的是 convolutional neural network 模型，CNN 
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
import os
import json
import gzip

# 使用结巴分词
# pip install jieba
import jieba
import jieba.analyse

curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")
model_dir = os.path.join(curr_dir, "cnn_model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
lex_file = os.path.join(curr_dir, "lex.pklz")
dataset_file = os.path.join(curr_dir, "dataset.pklz")

# 所有的打星
stars=["allstar10","allstar20","allstar30","allstar40","allstar50"]
# 打星转向量
def star_to_vector(star):
    v_star = np.zeros(len(stars))
    v_star[stars.index(star)] = 1
    return v_star
# 向量转打星
def vector_to_star(vec):
    char_pos = np.where(vec==np.max(vec))[-1][-1] 
    return stars[char_pos]

# lex:词汇表； comment:评论； star:评论对应的打分 
def comment_to_vector(lex, comment, star=None):
    words = jieba.cut(comment)
    features = np.zeros(len(lex))
    isBlank = True  #是否是无意义的评论，一个关键词都没有
    for word in words:
        if word in lex:
            isBlank = False
            features[lex.index(word)] = 1  # += 1 重复计数貌似没有多大的意义，反而是一个干扰项，因为算法中并没有考虑这个权重
    if isBlank:
        return None
    if star==None:
        return [features]
    return [features, star_to_vector(star)]

# 创建词汇表
# lex里保存了文本中出现过的单词。
def load_lex():
    if os.path.exists(lex_file):
        print("loading lex ...")
        lex = pickle.load(gzip.open(lex_file,"rb"))
    else:
        movies_file = os.path.join(data_dir,"movies.json")
        movies = json.loads(open(movies_file).read())
        def create_lexicon():    
            comments=""
            for title,_,_ in movies:
                movie_file =os.path.join(data_dir,u"{}.json".format(title))
                movie = json.loads(open(movie_file).read())                
                for _,comment in movie:
                    comments += comment
                    comments += '\n'
            # 获得权重最大的30000个单词
            return jieba.analyse.extract_tags(comments,topK=30000)
        lex = create_lexicon()
        with gzip.open(lex_file, 'wb') as f:
            pickle.dump(lex, f)
    print("lex",len(lex))
    return lex

lex = load_lex()

def load_dataset():
    if os.path.exists(dataset_file):
        print("loading dataset ...")
        dataset = pickle.load(gzip.open(dataset_file,"rb"))
    else:
        movies_file = os.path.join(data_dir,"movies.json")
        movies = json.loads(open(movies_file).read())

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
                    if one!=None:
                        dataset.append(one)   
            return dataset
        dataset = normalize_dataset(lex)
        random.shuffle(dataset)

        with gzip.open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)

    print("dataset",len(dataset))
    return dataset


# 定义每个层有多少'神经元''
lex_length = len(lex)
n_output_layer = len(stars)       # 输出层

X = tf.placeholder('int32', [None, lex_length]) # 输入值
Y = tf.placeholder('float')                     # 训练结果输入
dropout_keep_prob = tf.placeholder(tf.float32)

# 定义待训练的神经网络
def neural_network():
    embedding_size = 8 
    W = tf.Variable(tf.random_uniform([lex_length, embedding_size], -1.0, 1.0))  # (30000, 8)
    embedded_chars = tf.nn.embedding_lookup(W, X)   # (?, 30000) ==> (?, 30000, 8)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)    # (?, 30000, 8, 1)
    num_filters = 32
    filter_sizes = [3,4,5]
    pooled_outputs = []

    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]    # (3, 8, 1, 32)
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))  # (3, 8, 1, 32)
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))          # (32)
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID")  # (?, 29998, 1, 32)  # 29998 -> 29997 -> 29996
            h = tf.nn.relu(tf.nn.bias_add(conv, b))                                                 # (?, 29998, 1, 32)
            pooled = tf.nn.max_pool(h, ksize=[1, lex_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')    # (?, 1, 1, 32)
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)         # 32 * 3 = 96
    h_pool = tf.stack(pooled_outputs,axis=3)                    # (?, 1, 1, 3, 32)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])   # (?, 96)
    # dropout 繁殖过度拟合，可以不加 训练的时候选 0.5 测试的时候选 1
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    # output
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, n_output_layer], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[n_output_layer]))
        output = tf.nn.xw_plus_b(h_drop, W, b)		
    return output

predict = neural_network()  # 计算出来的输出结果
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))   # 损失函数
optimizer_nn = tf.train.AdamOptimizer(1e-3) # 优化器
grads_and_vars = optimizer_nn.compute_gradients(cost_func)
optimizer = optimizer_nn.apply_gradients(grads_and_vars)

# 使用数据训练神经网络
def train_neural_network(session):
    dataset = load_dataset()
    # 取样本中的10%做为测试数据
    test_size = int(len(dataset) * 0.1)
    dataset = np.array(dataset)
    train_dataset = dataset[:-test_size]
    test_dataset = dataset[-test_size:]

    batch_size = 20     # 每次使用 20 条数据进行训练，模型太大，系统跑不动
    epochs = 20         # 训练 10 轮
    random.shuffle(train_dataset)
    train_x = dataset[:, 0]
    train_y = dataset[:, 1]
    for epoch in range(epochs):
        epoch_loss = 0
        i = 0
        while i < len(train_x):
            start = i
            end = i + batch_size
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]
            _, c = session.run([optimizer, cost_func], feed_dict={X:list(batch_x), Y:list(batch_y), dropout_keep_prob:0.5})
            epoch_loss += c
            i += batch_size
            print(i, c)
        print(epoch, epoch_loss)

    text_x = test_dataset[:, 0]
    text_y = test_dataset[:, 1]
    correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    print(u'准确率: ', accuracy.eval({X:list(text_x) , Y:list(text_y) , dropout_keep_prob:1.0}))

    saver = tf.train.Saver(max_to_keep=1)
    saver_prefix = os.path.join(model_dir, "model.ckpt")
    saver.save(session, saver_prefix)

def test_neural_network(session):
    comment = raw_input("input comment:")        
    while len(comment)>0:
        x = comment_to_vector(lex, comment)
        if x!=None:
            y = session.run([predict],feed_dict={X:x})
            print(vector_to_star(y[0][0]))
        else:
            print(u"关键词不足，无法识别")    
        comment = raw_input("input comment:")

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver(max_to_keep=1)
    if ckpt and ckpt.model_checkpoint_path:
        print("restore model ...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("start train ...")    
        train_neural_network(sess)
    test_neural_network(sess)