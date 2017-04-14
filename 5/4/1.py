# coding=utf-8
import os
import pickle
import gzip
import collections
import random
import numpy as np
import tensorflow as tf


curr_dir = os.path.dirname(__file__)
data_dir = os.path.join(curr_dir, "data")
model_dir = os.path.join(curr_dir, "rnn_model")

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
poetrys_file = os.path.join(curr_dir, "poetrys.pklz")
words_file = os.path.join(curr_dir, "words.pklz")

# 装载诗句
def load_poetrys():
    if os.path.exists(poetrys_file):
        print("loading poetrys ...")
        poetrys = pickle.load(gzip.open(poetrys_file,"rb"))
    else:
        poetrys = []
        files = os.listdir(data_dir)
        for f in files:
            with open(os.path.join(data_dir,f),"r", encoding='utf-8') as lines:
                for line in lines:
                    line = line.replace("。",";")
                    line = line.replace("，",";")
                    line = line.replace("？",";")
                    line = line.replace("！",";")
                    line = line.replace("：",";")
                    line = line.replace(" ","")
                    for poetry in line.split(";"):
                        if len(poetry)==5 or len(poetry)==7:
                            poetrys.append('['+poetry+']')
        with gzip.open(poetrys_file, 'wb') as f:
            pickle.dump(poetrys, f)
    print("poetrys",len(poetrys))
    return poetrys                            

poetrys=load_poetrys()    
for i in range(10):
    print(poetrys[i])

# 获得诗歌字符和序号的map
def load_words():
    if os.path.exists(words_file):
        print("loading words ...")
        words = pickle.load(gzip.open(words_file,"rb"))
    else:
        all_words = []
        for poetry in poetrys:
            all_words += [word for word in poetry]
        counter = collections.Counter(all_words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)
        words = words[:len(words)] + (' ',) # 多个空格作为代替
        words = dict(zip(words, range(len(words))))    
        with gzip.open(words_file, 'wb') as f:
            pickle.dump(words, f)
    print("words",len(words))
    return words, sorted(words, key=words.get)  

words_map, words=load_words()
print(" ".join(words[:10]))
to_num = lambda word: words_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]

# 随机获得一批训练数据
def get_batch(batch_size):
    batches=random.sample(poetrys_vector, batch_size)
    length = max(map(len,batches))
    xdata = np.full((batch_size,length), words_map[' '], np.int32)   # 先全部填充空格
    for row in range(batch_size):
        xdata[row,:len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:,:-1] = xdata[:,1:]  # ？ydata 的前面有X的第一列开始覆写，最终重复了最后一列
    # print(xdata[0],ydata[0])
    return xdata, ydata

# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2, batch_size=128):
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])

    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell
 
    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
 
    initial_state = cell.zero_state(batch_size, tf.float32)
 
    with tf.variable_scope('rnnlm'):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words_map)+1])
        softmax_b = tf.get_variable("softmax_b", [len(words_map)+1])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [len(words_map)+1, rnn_size])
            inputs = tf.nn.embedding_lookup(embedding, input_data)
 
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs,[-1, rnn_size])
 
    logits = tf.matmul(output, softmax_w) + softmax_b
    probs = tf.nn.softmax(logits)
    return input_data, output_targets, logits, last_state, probs, cell, initial_state

def restore(sess):
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    saver_prefix = os.path.join(model_dir, "model.ckpt")        
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver(max_to_keep=1)
    if ckpt and ckpt.model_checkpoint_path:
        print("restore model ...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver, saver_prefix

#训练
def train_neural_network():
    batch_size = 128
    input_data, output_targets, logits, last_state, _, _, _ = neural_network(batch_size=batch_size)
    targets = tf.reshape(output_targets, [-1])
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
    cost = tf.reduce_mean(loss)
    # learning_rate = tf.Variable(0.0, trainable=False)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.0001, global_step, 10000, 0.97, staircase=True)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
 
        saver,checkpoint_path = restore(sess)
        # epoch = sess.run(global_step)
        # for epoch in range(50):
        while True:
            # sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            x_batch, y_batch = get_batch(batch_size)
            train_loss, epoch, _ , _ = sess.run([cost, global_step, last_state, train_op], feed_dict={input_data: x_batch, output_targets: y_batch})
            if epoch % 100 == 0:
                print(epoch, train_loss)
                saver.save(sess, checkpoint_path, global_step=epoch)

def to_word(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1)*s))
    return words[sample]

def gen_poetry():
    batch_size = 1
    input_data, output_targets, _, last_state, probs, cell, initial_state = neural_network(batch_size=batch_size)
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
 
        saver,checkpoint_path = restore(sess)
        state_ = sess.run(cell.zero_state(1, tf.float32))
 
        x = np.array([list(map(words_map.get, '['))])
        [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
        word = to_word(probs_)
        #word = words[np.argmax(probs_)]
        poem = ''
        while word != ']':
            poem += word
            x = np.zeros((1,1))
            x[0,0] = words_map[word]
            [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
            word = to_word(probs_)
            #word = words[np.argmax(probs_)]
        return poem
 
def gen_poetry_with_head(head):
    batch_size = 1
    input_data, output_targets, _, last_state, probs, cell, initial_state = neural_network(batch_size=batch_size)
 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
 
        saver,checkpoint_path = restore(sess)
 
        state_ = sess.run(cell.zero_state(1, tf.float32))
        poem = ''
        i = 0
        for word in head:
            while word != '，' and word != '。':
                poem += word
                x = np.array([list(map(words_map.get, word))])
                [probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
                word = to_word(probs_)
                time.sleep(1)
            if i % 2 == 0:
                poem += '，'
            else:
                poem += '。'
            i += 1
        return poem
 

if __name__ == '__main__':
    train_neural_network()
    print(gen_poetry())
   # print(gen_poetry_with_head('一二三四')) 
    
    