# coding=utf-8
# 中文OCR学习

import tensorflow as tf
import numpy as np
import os
from utils import readImgFile, img2bwinv, img2vec, dropZeroEdges, resize, save
import time
import random

curr_dir = os.path.dirname(__file__)

image_height = 28

#LSTM
num_hidden = 128
num_layers = 1

# 所有 unicode CJK统一汉字（4E00-9FBB） + ascii的字符加 + blank + ctc blank
# https://zh.wikipedia.org/wiki/Unicode
# https://zh.wikipedia.org/wiki/ASCII
ASCII_CHARS = [chr(c) for c in range(32,126+1)]
ZH_CHARS = [chr(c) for c in range(int('4E00',16),int('9FBB',16)+1)]
ZH_CHARS_PUN = ['。','？','！','，','、','；','：','「','」','『','』','‘','’','“','”',\
                '（','）','〔','〕','【','】','—','…','–','．','《','》','〈','〉']

CHARS = ASCII_CHARS + ZH_CHARS + ZH_CHARS_PUN
num_classes = len(CHARS) + 1 + 1

#初始化学习速率
# LEARNING_RATE_INITIAL = 1e-3
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 500
# MOMENTUM = 0.9

BATCHES = 64
BATCH_SIZE = 16
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = 10

train_files = []
if os.path.exists(os.path.join(curr_dir, "data", "index.txt")):
    print("Loading data ...")
    # 预处理图片
    if not os.path.exists(os.path.join(curr_dir, "dataset")):
        os.mkdir(os.path.join(curr_dir, "dataset"))
    with open(os.path.join(curr_dir, "data", "index.txt")) as index_file:
        for i, line in enumerate(index_file.readlines()):
            if i%10000==0: print("resizing image no: ",i)
            lines = line.split(" ")
            image_name = lines[0]+".png"
            dst_image_name = os.path.join(curr_dir,"dataset",image_name)
            if os.path.exists(dst_image_name):
                continue
            if not os.path.exists(os.path.dirname(dst_image_name)):
                os.mkdir(os.path.dirname(dst_image_name))        
            src_image_name = os.path.join(curr_dir,"data",image_name)
            try:
                image = readImgFile(src_image_name)
                image = img2bwinv(image)    
                image = dropZeroEdges(image)    
            except:
                print(dst_image_name,"error")
                continue
            resized_image = resize(image,image_height)
            save(resized_image,dst_image_name)
            train_files.append(line)


def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,12]
    inputs = tf.placeholder(tf.float32, [None, None, image_height], name="inputs")
    # 定义 ctc_loss 是稀疏矩阵
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    # 定义 LSTM 网络
    # 可以为:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
    cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=input_keep_prob)
    stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    
    # 第二个输出状态，不会用到
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)

    batch_s, max_timesteps = shape[0], shape[1]
    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    logits = tf.transpose(logits, (1, 0, 2), name="logits")
    return logits, inputs, labels, seq_len, W, b, input_keep_prob


# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch(batch_size=128):
    batch = random.sample(train_files, batch_size)    
    codes = []
    images = []
    max_width_image = 0
    for line in batch:
        lines = line.split(" ")
        imageFileName = lines[0]+".png"
        text = line[line.index(' '):].strip()
        # 在宋体9号字体下，O和0完全一致，因此全部按0处理
        # text = text.replace('O','0')
        # 输出图片为反色黑白
        image = readImgFile(os.path.join(curr_dir,"dataset",imageFileName))    
        images.append(image)
        if image.shape[0] > max_width_image: 
            max_width_image = image.shape[0]
        text_list = [CHARS.index(char) for char in text]
        codes.append(text_list)

    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = img2vec(image, width=max_width_image, height=image_height)
        inputs[i,:] = np.transpose(image_vec.reshape((image_height,max_width_image)))

    labels = [np.asarray(i) for i in codes]
    #labels转成稀疏矩阵
    sparse_labels = sparse_tuple_from(labels)
    seq_len = np.ones(inputs.shape[0]) * max_width_image
    return inputs, sparse_labels, seq_len

# 转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result
    
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = spars_tensor[1][m]
        decoded.append(str)
    return decoded

def list_to_chars(list):
    return "".join([CHARS[v] for v in list])

def train():
    global_step = tf.Variable(0, trainable=False)
    
    # learning_rate = tf.train.exponential_decay(LEARNING_RATE_INITIAL,
                                            #    global_step,
                                            #    LEARNING_RATE_DECAY_STEPS,
                                            #    LEARNING_RATE_DECAY_FACTOR,
                                            #    staircase=True, name="learning_rate")
    # 决定还是自定义学习速率比较靠谱                                            
    curr_learning_rate = 1e-3
    learning_rate = tf.placeholder(tf.float32, shape=[])                                            

    logits, inputs, labels, seq_len, W, b, input_keep_prob = neural_networks()

    loss = tf.nn.ctc_loss(labels=labels,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss, name="cost")

    # 收敛效果不好
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step, name="optimizer")
    # 直接最小化 loss 容易过拟合
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels), name="acc")

    init = tf.global_variables_initializer()

    def report_accuracy(decoded_list, test_labels):
        original_list = decode_sparse_tensor(test_labels)
        detected_list = decode_sparse_tensor(decoded_list)
        true_numer = 0
        
        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
            return
        print("T/F: original(length) <-------> detectcted(length)")
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            print(hit, list_to_chars(number), "(", len(number), ") <-------> ", list_to_chars(detect_number), "(", len(detect_number), ")")
            if hit:
                true_numer = true_numer + 1
        print("Test Accuracy:", true_numer * 1.0 / len(original_list))

    def do_report():
        test_inputs,test_labels,test_seq_len = get_next_batch(TEST_BATCH_SIZE)
        test_feed = {inputs: test_inputs,
                     labels: test_labels,
                     seq_len: test_seq_len,
                     input_keep_prob: 1.0}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        report_accuracy(dd, test_labels)
 
    def do_batch():
        train_inputs, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)       
        feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len,
                input_keep_prob: 0.7, learning_rate: curr_learning_rate}        
        b_loss,b_labels, b_logits, b_seq_len,b_cost, steps, b_learning_rate, _ = session.run([loss, labels, logits, seq_len, cost, global_step, learning_rate, optimizer], feed)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()