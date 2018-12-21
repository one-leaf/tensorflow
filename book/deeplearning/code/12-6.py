# 英译汉，翻译
import os, json, tarfile
import urllib.request
import jieba
import jieba.analyse
import tensorflow as tf
import numpy as np
import math, random

curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(curr_path,"../data")
logs_path = os.path.join(curr_path,"../logs")
if not os.path.exists(data_path): os.makedirs(data_path)
if not os.path.exists(logs_path): os.makedirs(logs_path)

# 定义wmt18数据集，前面为训练数据，后面为测试数据
dataset_files = {"training-parallel-nc-v13.tgz":
                        ["training-parallel-nc-v13/news-commentary-v13.zh-en.en",
                         "training-parallel-nc-v13/news-commentary-v13.zh-en.zh"],
                 "dev.tgz":["dev/newsdev2017-enzh-src.en.sgm",
                            "dev/newsdev2017-enzh-ref.zh.sgm"]} 

# 获取 训练和测试数据
def downloadData():
    for dataset_file in dataset_files:
        loc_file =  os.path.join(data_path, dataset_file)
        if not os.path.exists(loc_file) :  
            print("Download Dataset File %s ..."%dataset_file)
            urllib.request.urlretrieve ("http://data.statmt.org/wmt18/translation-task/"+dataset_file, loc_file)
        for extract_file in dataset_files[dataset_file]:
            loc_extract_file = os.path.join(data_path, extract_file)
            if not os.path.exists(loc_extract_file):
                tar = tarfile.open(loc_file)
                tar.extract(extract_file, path=data_path)
                tar.close()

class word2vec_network():
    def __init__(self, embedding_size, batch_size, words, sentences_file):
        vocabulary_size = len(words)
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
                                            num_sampled=batch_size//2,
                                            num_classes=vocabulary_size))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(self.loss)       
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
        self.normalized_embeddings = self.embeddings / norm

        self.words = words
        self.sentences_file = sentences_file
        self.batch_size = batch_size

    def generate_batch(self, skip_window = 1, num_skips = 2):    
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        batch_index = 0
        span = 2 * skip_window + 1   
        assert num_skips <= 2 * skip_window
        for s in open(self.sentences_file, encoding="UTF8").readlines():
            data=[] # 词列表
            s = s.lower()
            _words = jieba.lcut(s)
            for word in _words:
                word = word.strip()
                if word == '': continue
                if word in self.words:
                    data.append(self.words[word])
                else:
                    data.append(self.words['UNK'])
            if len(data)<span: continue
            for i in range(len(data)-span+1):
                buffer = data[i:i+span]
                context_words = [w for w in range(span) if w != skip_window]
                words_to_use = random.sample(context_words, num_skips)
                for j, context_word in enumerate(words_to_use):
                    batch[batch_index] = buffer[skip_window]
                    labels[batch_index, 0] = buffer[context_word]
                    batch_index += 1               
                    if batch_index == self.batch_size:
                        yield batch, labels
                        batch_index = 0  

    def train(self, epoch_number):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(epoch_number):
                for step, (batch, labels) in enumerate(self.generate_batch(1, 2)):
                    loss,_= sess.run([self.loss, self.optimizer], feed_dict={self.x: batch, self.y: labels})
                    if step % 1000 == 0:
                        print(epoch, "/", epoch_number, step, loss)
            return self.normalized_embeddings.eval()

# 输出词向量和词典
def word2vec(sentences_file, words_filename, words_embedding_filename, words_number=5000, embedding_size=32*32, epoch_number=100):
    save_words_dict_file = os.path.join(logs_path, words_filename)
    if not os.path.exists(save_words_dict_file):
        print("Create", words_filename, "...")
        text = open(sentences_file, encoding="UTF8").read()
        text = text.lower()
        text = text.replace("\n"," ")
        words=jieba.analyse.extract_tags(text, topK=words_number-1)
        words_dict={"UNK":0}
        for i,word in enumerate(words):
            words_dict[word]=i+1
        save_words_dict_file = open(save_words_dict_file,"w",encoding="UTF8")
        json.dump(words_dict, save_words_dict_file, ensure_ascii=False)
    else:
        save_words_dict_file = open(save_words_dict_file,"r",encoding="UTF8")
        words_dict=json.load(save_words_dict_file)

    save_words_embedding_filename = os.path.join(logs_path, words_embedding_filename)   
    if not os.path.exists(save_words_embedding_filename):
        batch_size = 64
        net = word2vec_network(embedding_size, batch_size, words_dict, sentences_file)
        print("start traing", words_embedding_filename, '...')
        final_embeddings = net.train(epoch_number)
        np.save(save_words_embedding_filename, final_embeddings)

class translate_network():
    def __init__(self, words, embedding_size, batch_size):
        vocabulary_size = len(words)
        self.en = tf.placeholder(tf.int32, [None, None])
        self.en_seq_len = tf.placeholder(tf.int32, [None],
        self.zh = tf.placeholder(tf.int32, [None, None])
        self.zh_seq_len = tf.placeholder(tf.int32, [None],
        self.en_embedding = tf.Variable(tf.zeros([vocabulary_size, embedding_size]), dtype=tf.float32)
        self.zh_embedding = tf.Variable(tf.zeros([vocabulary_size, embedding_size]), dtype=tf.float32)
        self.en_vec = tf.nn.embedding_lookup(self.en_embedding, self.en)
        self.zh_vec = tf.nn.embedding_lookup(self.zh_embedding, self.zh)
    
        # encode
        num_hidden = 128
        with tf.variable_scope('ENCODE_EN_RNN'):
            cell_fw = tf.contrib.rnn.GRUCell(num_hidden//2)
            cell_bw = tf.contrib.rnn.GRUCell(num_hidden//2)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.en_vec, seq_len, dtype=tf.float32)
            en_layer = tf.concat(outputs, axis=-1)


          

def main():
    # 下载文件
    downloadData()

    words_number=5000
    embedding_size=32*32

    # 训练并产生 embedding 文件
    for sfile in dataset_files["training-parallel-nc-v13.tgz"]:
        sentences_file = os.path.join(data_path, sfile)
        words_filename = "%s_words.json"%sfile[-2:]
        words_embedding_filename = "%s_embedding.npy"%sfile[-2:]
        word2vec(sentences_file, words_filename, words_embedding_filename,
                    words_number=words_number, embedding_size=embedding_size, epoch_number=1)

    # 

if __name__ == "__main__":
    main()