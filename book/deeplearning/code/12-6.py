# 英译汉，翻译
import os, json, tarfile
import urllib.request
import jieba
import jieba.analyse
import tensorflow as tf
import numpy as np
import math, random
import collections

curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(curr_path,"../data")
logs_path = os.path.join(curr_path,"../logs")
if not os.path.exists(data_path): os.makedirs(data_path)
if not os.path.exists(logs_path): os.makedirs(logs_path)

# 定义wmt18数据集，前面为训练数据，后面为测试数据
dataset_files = {"train": { "source": "training-parallel-nc-v13.tgz", 
                            "files": {
                                "en": "training-parallel-nc-v13/news-commentary-v13.zh-en.en",
                                "zh": "training-parallel-nc-v13/news-commentary-v13.zh-en.zh"}
                            },
                "valid": { "source": "dev.tgz",
                            "files": {
                                "en":"dev/newsdev2017-enzh-src.en.sgm",
                                "zh":"dev/newsdev2017-enzh-ref.zh.sgm"}
                            }
                }

# 定义常量 PAD : 补全， SOS (start of sentence) 翻译开始， EOS (end of sentence) 翻译结束， UNK (unknown) 未知单词
KEY_WORDS = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}

# 获取 训练和测试数据
def downloadData():
    for key in dataset_files:
        dataset_file = dataset_files[key]["source"]
        loc_file =  os.path.join(data_path, dataset_file)
        if not os.path.exists(loc_file) :  
            print("Download Dataset File %s ..."%dataset_file)
            urllib.request.urlretrieve ("http://data.statmt.org/wmt18/translation-task/"+dataset_file, loc_file)
        for lang in dataset_files[key]["files"]:
            extract_file = dataset_files[key]["files"][lang]
            loc_extract_file = os.path.join(data_path, extract_file)
            if not os.path.exists(loc_extract_file):
                tar = tarfile.open(loc_file)
                tar.extract(extract_file, path=data_path)
                tar.close()

# 输出词典
def load_word_dict(sentences_file, words_filename, words_number=5000):
    save_words_dict_file = os.path.join(logs_path, words_filename)
    if not os.path.exists(save_words_dict_file):
        print("Create", words_filename, "...")
        text = open(sentences_file, encoding="UTF8").read()
        text = text.lower()
        text = text.replace("\n"," ")
        words = jieba.lcut(text)
        # 移除空格
        for _ in xrange(words.count(' ')): words.remove(' ') 
        words = collections.Counter(words).most_common(words_number-len(KEY_WORDS))
        words_dict={}
        for word in KEY_WORDS:
            words_dict[word] = KEY_WORDS[word]
        n = len(KEY_WORDS)
        for i, (word, _) in enumerate(words):
            words_dict[word] = i + n
        save_words_dict_file = open(save_words_dict_file,"w",encoding="UTF8")
        json.dump(words_dict, save_words_dict_file, ensure_ascii=False)
    else:
        save_words_dict_file = open(save_words_dict_file,"r",encoding="UTF8")
        words_dict=json.load(save_words_dict_file)
    return words_dict

# 输出句子
def load_sentences(sentences_file, words_dict, sentences_vec_file):
    save_sentences_vec_file = os.path.join(logs_path, sentences_vec_file)
    sentences_vec=[]
    if not os.path.exists(save_sentences_vec_file):
        print("Create", sentences_vec_file, "...")
        for line in open(sentences_file, encoding="UTF8").readlines():
            text = line.lower()
            text = text.replace("\n","")
            words=jieba.lcut(text)
            # 移除空格
            for _ in xrange(words.count(' ')): words.remove(' ') 
            s_vec = []
            for w in words:
                if w in words_dict:
                    s_vec.append(words_dict[w])
                else:
                    s_vec.append(KEY_WORDS["<UNK>"])
            sentences_vec.append(s_vec)
        json.dump(sentences_vec, open(save_sentences_vec_file,"w",encoding="UTF8"))
    else:
        sentences_vec = json.load(open(save_sentences_vec_file,"r",encoding="UTF8"))
    return sentences_vec


# 翻译seq2seq网络
# 原理
# how are you <SOS> 你 好 吗 <PAD> --> encode --> decode --> 你 好 吗 <EOS> <PAD>
# 其中 <SOS> --> 你 ， 你 --> 好 , 好 --> 吗， 吗 --> <EOS>
class translate_network():
    # 多层 rnn 模型
    def get_rnn_cell(self, run_size, rnn_layers_num):
        return tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(rnn_size) for _ in range(rnn_layers_num)])

    def __init__(self, en_words, zh_words, embedding_size, rnn_size=128, rnn_layers_num=3, batch_size=64):
        en_vocab_size = len(en_words)                                      # 英语词汇表长度
        zh_vocab_size = len(zh_words)                                      # 中文词汇表长度
        self.en = tf.placeholder(tf.int32, [None, None])                        # 英文句子 [batch_size, sentence]
        self.en_seq_len = tf.placeholder(tf.int32, [None])                      # 英文句子的长度 [batch_size]
        self.zh = tf.placeholder(tf.int32, [None, None])                        # 中文句子 [batch_size, sentence]
        self.zh_seq_len = tf.placeholder(tf.int32, [None])                      # 中文句子的长度 [batch_size]
        # 英文词嵌入向量表 [batch_size vocab_size embed_dim] 这里可以用word2vec
        self.en_embedding = tf.contrib.layers.embed_sequence(self.en, en_vocab_size, embedding_size) 
        # 中文词嵌入向量表 [batch_size vocab_size embed_dim] 这里可以用word2vec      
        self.zh_embedding = tf.contrib.layers.embed_sequence(self.zh, zh_vocab_size, embedding_size)       
       
        # encoder 编码器
        encoder_cell = self.get_rnn_cell(rnn_size, rnn_layers_num)
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.en_embedding, self.en_seq_len, dtype=tf.float32)

        # attention 注意力
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=rnn_size, memory=encoder_outputs)
        decoder_cell = self.get_rnn_cell(rnn_size, rnn_layers_num)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=rnn_size)

        # 定义 训练 helper
        helper = tf.contrib.seq2seq.TrainingHelper(self.zh_embedding, self.zh_seq_len)
        
        # decoder 解码器
        projection_layer = tf.layers.Dense(zh_vocabulary_size, use_bias=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, projection_layer)

        # 定义最长翻译输出为输入的20倍
        maximum_iterations = tf.round(tf.reduce_max(self.en_seq_len) * 20)
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        self.logits = outputs.rnn_output

        # 定义预测输出 Infer
        # 定义 预测 helper ，这里的 zh_embedding 为 <STR> zh_seq_len 为 [batch_size] 值为 KEY_WORS["<SOS>"] 表示翻译开始
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.zh_embedding, self.zh_seq_len, KEY_WORDS["<EOS>"])         
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, infer_helper, encoder_state, projection_layer)
        infer_outpus, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
        self.translations = infer_outpus.sample_id

        targets = tf.reshape(self.zh, [-1])
        logits_flat = tf.reshape(logits, [-1, zh_vocabulary_size])
        self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

def translate():
    pass

def main():
    # 下载文件
    downloadData()

    embedding_size=32*32
    en_words_number=80000
    zh_words_number=50000

    # 训练并产生 embedding 文件
    # 源语言
    en_sentences_file = os.path.join(data_path, dataset_files["train"]["files"]["en"])
    en_words_dict = load_word_dict(en_sentences_file, "en_words.json", words_number=en_words_number)
    en_sentences_vec = load_sentences(en_sentences_file, en_words_dict, "en_sentences_vec.json")
    # 目标语言
    zh_sentences_file = os.path.join(data_path, dataset_files["train"]["files"]["zh"])
    zh_words_dict = load_word_dict(zh_sentences_file, "zh_words.json", words_number=zh_words_number)
    zh_sentences_vec = load_sentences(zh_sentences_file, zh_words_dict, "zh_sentences_vec.json")


    # 训练SEQ-SEQ网络
    translate()

if __name__ == "__main__":
    main()