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
        print("create", words_filename, "...")
        words =[]
        for text in open(sentences_file, encoding="UTF8").readlines():
            text = text.lower()
            text = text.replace("\n","")
            _words = jieba.lcut(text)
            for _ in range(_words.count(' ')): _words.remove(' ')
            words += _words
             
        words = collections.Counter(words)
        print(words_filename, len(words))
        words = words.most_common(words_number-len(KEY_WORDS))
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
    print(words_filename, len(words_dict))
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
            for _ in range(words.count(' ')): words.remove(' ') 
            s_vec = []
            for w in words:
                s_vec.append(words_dict.get(w, KEY_WORDS["<UNK>"]))
            sentences_vec.append(s_vec)
        json.dump(sentences_vec, open(save_sentences_vec_file,"w",encoding="UTF8"))
    else:
        sentences_vec = json.load(open(save_sentences_vec_file,"r",encoding="UTF8"))
    print(sentences_vec_file, len(sentences_vec))
    return sentences_vec


# 翻译seq2seq网络
# 原理
# how are you <SOS> 你  好  吗
# --> encode --> decode --> 
#               你  好  吗 <EOS> 
# 其中 <SOS> --> 你 ， 你 --> 好 , 好 --> 吗 ， 吗 --> <EOS>
class nmt_network():
    # 多层 rnn 模型
    def get_rnn_cell(self, rnn_size, rnn_layers_num):
        def single_cell():
            cell = tf.nn.rnn_cell.LSTMCell(rnn_size)
            keep_prob = tf.cond(self.is_training, lambda:tf.constant(0.8), lambda:tf.constant(1.0))
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, keep_prob) 
            return cell
        return tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(rnn_layers_num)])

    def __init__(self, en_words, zh_words, embedding_size, rnn_size=128, rnn_layers_num=2, beam_search=True, beam_size=10):
        self.en = tf.placeholder(tf.int32, [None, None])                        # 英文句子 [batch_size, sentence]
        self.en_seq_len = tf.placeholder(tf.int32, [None])                      # 英文句子的长度 [batch_size]
        self.zh = tf.placeholder(tf.int32, [None, None])                        # 中文句子 [batch_size, sentence] 前后有加 <SOS> <EOS>
        self.zh_seq_len = tf.placeholder(tf.int32, [None])                      # 中文句子的长度 [batch_size]
        self.is_training = tf.placeholder_with_default(True, shape=(), name='is_training')
        self.beam_search = beam_search              # 是否采用 beam_search 函数预测，从 beam_size 个最大概率后选中寻找
        self.beam_size = beam_size

        batch_size = tf.size(self.en_seq_len)
        en_vocab_size = len(en_words)                                      # 英语词汇表长度
        zh_vocab_size = len(zh_words)                                      # 中文词汇表长度

        self.zh_input  = self.zh[:, :-1]                # decoder 输入中文为 <SOS> + sentence
        self.zh_output = self.zh[:, 1:]                 # loss 输出中文为 sentence + <EOS>

        # 英文词嵌入向量表 [batch_size en_vocab_size embed_dim] 这里可以用word2vec
        self.en_embeddings = tf.Variable(tf.random_uniform([en_vocab_size, embedding_size])) 
        self.en_embedded = tf.nn.embedding_lookup(self.en_embeddings, self.en)
        # 中文词嵌入向量表 [batch_size zh_vocab_size embed_dim] 这里可以用word2vec      
        self.zh_embeddings = tf.Variable(tf.random_uniform([zh_vocab_size, embedding_size])) 
        self.zh_embedded = tf.nn.embedding_lookup(self.zh_embeddings, self.zh_input)

        print("en_embedded", self.en_embedded)
        print("zh_embedded", self.zh_embedded)

        print("-"*100,"encoder")
        # encoder 编码器
        with tf.variable_scope('encoder'):
            fw_cell = self.get_rnn_cell(rnn_size, rnn_layers_num)
            bw_cell = self.get_rnn_cell(rnn_size, rnn_layers_num)
            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.en_embedded, self.en_seq_len, dtype=tf.float32)
            encoder_outputs = tf.concat(bi_outputs, -1)         # [batch_size en_vocab_size rnn_size*2]
            encoder_state = []
            for i in range(rnn_layers_num):
                encoder_state.append(bi_state[0][i])
                encoder_state.append(bi_state[1][i])
            encoder_state = tuple(encoder_state)                # rnn_layers_num*2*[batch_size rnn_size]
            print('encoder_outputs', encoder_outputs)
            print('encoder_state', encoder_state)

        print("-"*100,"decoder")
        with tf.variable_scope('decoder'):
            if self.beam_search:
                print("use beamsearch decoding ...")
                memory = tf.contrib.seq2seq.tile_batch(encoder_outputs, self.beam_size)          # [batch_size*beam_size en_vocab_size rnn_size*2]
                memory_state = tf.contrib.seq2seq.tile_batch(encoder_state, self.beam_size)    # rnn_layers_num*[batch_size*beam_size rnn_size]
                x_len = tf.contrib.seq2seq.tile_batch(self.en_seq_len, self.beam_size)  # [batch_size*beam_size]
                bs = batch_size * self.beam_size                                        
            else:
                memory = encoder_outputs
                memory_state = encoder_state
                x_len = self.en_seq_len
                bs = batch_size

            # 如果是训练阶段，则采用原生的
            memory = tf.cond(self.is_training, lambda:encoder_outputs, lambda:memory)
            memory_state = tf.cond(self.is_training, lambda:encoder_state, lambda:memory_state)
            x_len = tf.cond(self.is_training, lambda:self.en_seq_len, lambda:x_len)
            bs = tf.cond(self.is_training, lambda:batch_size, lambda:bs)

            print("memory", memory)
            print("memory_state", memory_state)
            print("x_len", x_len)

            # multiplicative attention 注意力乘法模式
            # attention = tf.contrib.seq2seq.LuongAttention(rnn_size, memory, x_len) 
            # additive attention 注意力加法模式
            attention = tf.contrib.seq2seq.BahdanauAttention(rnn_size, memory, x_len)
            print('attention', attention) 

            # 由于encoder采用了双向RNN，所以这里需要*2
            decoder_cell = self.get_rnn_cell(rnn_size, rnn_layers_num*2)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention, rnn_size)
            print('decoder_cell', decoder_cell) 
            decoder_initial_state = decoder_cell.zero_state(bs, tf.float32).clone(cell_state=memory_state)
            print('decoder_initial_state', decoder_initial_state)

            # decoder 解码器
            projection_layer = tf.layers.Dense(zh_vocab_size, use_bias=False, kernel_initializer=tf.contrib.layers.xavier_initializer())
            print('projection_layer', projection_layer)
            # 定义最长翻译输出为输入长度的2倍
            maximum_iterations = tf.round(tf.reduce_max(self.en_seq_len) * 3)

            # 定义 训练 helper , 如何根据预测结果得到下一时刻的输入。
            # TrainingHelper 直接用上一时刻的真实值作为下一时刻的输入
            train_helper = tf.contrib.seq2seq.TrainingHelper(self.zh_embedded, self.zh_seq_len, time_major=False)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, decoder_initial_state, projection_layer)
            train_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder)
            self.logits = train_outputs.rnn_output  # [batch_size, zh_seq_len, vocab_size]
            print("logits", self.logits)

        print("-"*100,"infer")
        with tf.variable_scope("decoder", reuse=True):
            # 定义预测输出 Infer
            # start_tokens = tf.fill([batch_size], KEY_WORDS["<SOS>"])
            start_tokens = tf.ones([batch_size,],tf.int32) * KEY_WORDS["<SOS>"]
            print('start_tokens',start_tokens)
            if self.beam_search:
                infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell, self.zh_embeddings, 
                    start_tokens, KEY_WORDS["<EOS>"], decoder_initial_state, self.beam_size, projection_layer)
                print("infer_decoder",infer_decoder)
                infer_outpus, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder, maximum_iterations=maximum_iterations)
                self.translations = infer_outpus.predicted_ids  # [batch_size, zh_seq_len, beam_size]
            else:
                # 定义 预测 helper ，GreedyEmbeddingHelper 用当前概率最大的输出作为下一时刻的输入
                # 这里的 zh_embedding 为 <STR> zh_seq_len 为 [batch_size] 值为 KEY_WORS["<EOF>"] 表示翻译结束
                infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.zh_embeddings, start_tokens, KEY_WORDS["<EOS>"])   
                infer_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, infer_helper, decoder_initial_state, projection_layer)
                print("infer_decoder",infer_decoder)
                infer_outpus, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder, maximum_iterations=maximum_iterations)
                self.translations = tf.expand_dims(infer_outpus.sample_id, -1)  # [batch_size, zh_seq_len, 1]

            print("translations", self.translations)


        # 定义训练损失函数
        # targets = tf.reshape(self.zh, [-1])
        # logits_flat = tf.reshape(self.logits, [-1, zh_vocab_size])
        # self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)

        mask = tf.sequence_mask(self.zh_seq_len, tf.reduce_max(self.zh_seq_len), dtype=tf.float32)
        # logits_train = tf.argmax(self.logits, axis=-1)
        self.cost = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.zh_output, weights=mask)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5.0)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

def get_Data(en_sentences_vec, zh_sentences_vec, batch_size):
    en_len = len(en_sentences_vec)
    ids = random.sample(range(en_len), batch_size)

    # 抽取数据，加上前后开头
    en_sentences_vec_batch=[] 
    zh_sentences_vec_batch=[] 
    for id in ids:
        en_sentences_vec_batch.append(en_sentences_vec[id])
        zh_sentences_vec_batch.append([KEY_WORDS["<SOS>"]] + zh_sentences_vec[id] + [KEY_WORDS["<EOS>"]])

    # 确定输入序列长度
    en_sentences_vec_batch_lens = [len(sentence) for sentence in en_sentences_vec_batch]
    zh_sentences_vec_batch_lens = [len(sentence) for sentence in zh_sentences_vec_batch]

    # 求当前批次最大长度
    en_sentences_vec_batch_maxlen = max(en_sentences_vec_batch_lens)
    # 由于前面减少了一位，所以这里求最大长度需要加1
    zh_sentences_vec_batch_maxlen = max(zh_sentences_vec_batch_lens)

    # 按当前批次最大长度补齐
    en_sentences_vec_batch_pad = [sentence + [KEY_WORDS["<PAD>"]] * (en_sentences_vec_batch_maxlen - len(sentence)) for sentence in en_sentences_vec_batch]
    zh_sentences_vec_batch_pad = [sentence + [KEY_WORDS["<PAD>"]] * (zh_sentences_vec_batch_maxlen - len(sentence)) for sentence in zh_sentences_vec_batch]

    # 由于输入的中文decode会减少<EOF>，loss时，会减少<EOS>，所以总长减少一位
    zh_sentences_vec_batch_lens = [sentence_len-1 for sentence_len in zh_sentences_vec_batch_lens]
    return en_sentences_vec_batch_pad, en_sentences_vec_batch_lens, zh_sentences_vec_batch_pad, zh_sentences_vec_batch_lens

def train(en_sentences_vec, zh_sentences_vec, net, batch_size=30, epochs=10):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver, checkpoint_prefix = get_saver(sess)
        for e in range(epochs):
            for step in range(len(en_sentences_vec)//batch_size): 
                # 获得训练数据
                en_batch, en_batch_lens, zh_batch, zh_batch_lens = get_Data(en_sentences_vec, zh_sentences_vec, batch_size)
                # 训练
                _, loss = sess.run( [net.train_op, net.cost],
                    {net.en: en_batch, net.en_seq_len: en_batch_lens, 
                    net.zh: zh_batch, net.zh_seq_len: zh_batch_lens, 
                    net.is_training: True})  
                print(e, step, loss)  

                if step % 100 == 0:
                    saver.save(sess, checkpoint_prefix)
                    en_batch, en_batch_lens, zh_batch, zh_batch_lens = get_Data(en_sentences_vec, zh_sentences_vec, 2)
                    ids = sess.run( net.translations,
                        {net.en: en_batch, net.en_seq_len: en_batch_lens, 
                        net.is_training: False}) 
                    print("infer", ids[:, :, 0])
                    print("zh", zh_batch)

def get_saver(sess):
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
    checkpoint_prefix = os.path.join(logs_path, "12_6_model.ckpt")
    ckpt = tf.train.get_checkpoint_state(logs_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("restore checkpoint and continue train.")
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver, checkpoint_prefix

def main():
    # 下载文件
    downloadData()

    embedding_size  = 512       # 词向量维度
    en_words_number = 50000     # 共 53839 个单词
    zh_words_number = 90000     # 共 91436 个单词

    # 训练并产生 embedding 文件
    # 源语言
    en_sentences_file = os.path.join(data_path, dataset_files["train"]["files"]["en"])
    en_words_dict = load_word_dict(en_sentences_file, "en_words.json", words_number=en_words_number)
    en_sentences_vec = load_sentences(en_sentences_file, en_words_dict, "en_sentences_vec.json")
    # 目标语言
    zh_sentences_file = os.path.join(data_path, dataset_files["train"]["files"]["zh"])
    zh_words_dict = load_word_dict(zh_sentences_file, "zh_words.json", words_number=zh_words_number)
    zh_sentences_vec = load_sentences(zh_sentences_file, zh_words_dict, "zh_sentences_vec.json")

    net =  nmt_network(en_words_dict, zh_words_dict, embedding_size, rnn_size=512, beam_search=True)

    # 训练SEQ2SEQ网络   
    train(en_sentences_vec, zh_sentences_vec, net, batch_size=64, epochs=20)

if __name__ == "__main__":
    main()