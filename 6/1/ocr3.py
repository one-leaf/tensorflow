# coding=utf-8
# 中文OCR学习，尝试多层

import tensorflow as tf
import numpy as np
import os
from utils import readImgFile, img2bwinv, img2vec, dropZeroEdges, resize, save
import time
import random

curr_dir = os.path.dirname(__file__)

image_height = 16

# LSTM
num_hidden = 8
num_layers = 1

# 所有 unicode CJK统一汉字（4E00-9FBB） + ascii的字符加 + blank + ctc blank
# https://zh.wikipedia.org/wiki/Unicode
# https://zh.wikipedia.org/wiki/ASCII
ASCII_CHARS = [chr(c) for c in range(32,126+1)]
ZH_CHARS = [chr(c) for c in range(int('4E00',16),int('9FBB',16)+1)]
ZH_CHARS_PUN = ['。','？','！','，','、','；','：','「','」','『','』','‘','’','“','”',\
                '（','）','〔','〕','【','】','—','…','–','．','《','》','〈','〉']

CHARS = ASCII_CHARS + ZH_CHARS + ZH_CHARS_PUN
# CHARS = ASCII_CHARS
num_classes = len(CHARS) + 1 + 1

#初始化学习速率
# LEARNING_RATE_INITIAL = 1e-3
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 500
MOMENTUM = 0.9

BATCHES = 64
BATCH_SIZE = 64
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = 64

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
                train_files.append(line)
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
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, None, image_height], name="inputs")
    # 定义 ctc_loss 是稀疏矩阵
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    # 1维向量 size [batch_size] 等于 np.ones(batch_size)* image_width
    seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
    shape = tf.shape(inputs)
    batch_size, max_timesteps = shape[0], shape[1]

    # 第一种双向LSTM方法
    # cell_fw = tf.contrib.rnn.LSTMCell(num_hidden/2, state_is_tuple=True)
    # cell_bw = tf.contrib.rnn.LSTMCell(num_hidden/2, state_is_tuple=True)                       
    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
    # outputs = tf.concat(outputs, axis=2)
    # stack_cell = tf.contrib.rnn.MultiRNNCell(
    #             [tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True) for _ in range(num_layers)],
    #             state_is_tuple=True)
    # lstm_out, last_state = tf.nn.dynamic_rnn(stack_cell, outputs, seq_len, dtype=tf.float32)
    # lstm_out = tf.reshape(lstm_out, [-1, num_hidden])
    # W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
    # b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    # logits = tf.matmul(lstm_out, W) + b

    # 第二种双向LSTM方法,很容易全部为blank ？
    # cell_fw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    # cell_bw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
    # outputs_fw = tf.reshape(outputs[0], [-1, num_hidden])
    # outputs_bw = tf.reshape(outputs[1], [-1, num_hidden])
    # W_fw = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1, dtype=tf.float32))
    # W_bw = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1, dtype=tf.float32))
    # b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    # logits = tf.add(tf.matmul(outputs_fw, W_fw),tf.matmul(outputs_bw, W_bw)) + b

    # 第三种双向LSTM方法
    cell_fw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
    outputs = tf.concat(outputs, axis=2)
    outputs = tf.reshape(outputs, [-1, num_hidden*2 ])
    W = tf.Variable(tf.truncated_normal([num_hidden*2, num_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    logits = tf.matmul(outputs, W) + b   

    # logits = tf.nn.softmax(logits)
    # 输出对数： [batch_size , max_time , num_classes]
    logits = tf.reshape(logits, [batch_size, -1, num_classes])
    # 需要变换到 time_major == True [max_time x batch_size x num_classes]
    logits = tf.transpose(logits, (1, 0, 2), name="logits")
    return logits, inputs, labels, seq_len, input_keep_prob


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
        # 输入的图片为反色黑白
        image = readImgFile(os.path.join(curr_dir,"dataset",imageFileName))    
        images.append(image)
        if image.shape[1] > max_width_image: 
            max_width_image = image.shape[1]
        text_list = [CHARS.index(char) for char in text]
        codes.append(text_list)

    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = img2vec(images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.transpose(image_vec)

    labels = [np.asarray(i) for i in codes]
    #labels转成稀疏矩阵
    sparse_labels = sparse_tuple_from(labels)
    seq_len = np.ones(batch_size) * max_width_image
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
    curr_learning_rate = 1e-4
    learning_rate = tf.placeholder(tf.float32, shape=[])                                            

    logits, inputs, labels, seq_len, input_keep_prob = neural_networks()

    # If time_major == True (default), this will be a Tensor shaped: [max_time x batch_size x num_classes]
    # 返回 A 1-D float Tensor, size [batch], containing the negative log probabilities.
    loss = tf.nn.ctc_loss(labels=labels,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss, name="cost")

    # 收敛效果不好
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM).minimize(cost, global_step=global_step)

    # 做一个梯度裁剪，貌似也没啥用
    # grads_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # grads_and_vars = grads_optimizer.compute_gradients(loss)
    # capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
    # gradients, variables = zip(*grads_optimizer.compute_gradients(loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    # capped_grads_and_vars = zip(gradients, variables)

    #capped_grads_and_vars = [(tf.clip_by_norm(g, 5), v) for g,v in grads_and_vars]
    # optimizer = grads_optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)

    # 直接最小化 loss 容易过拟合
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    # The ctc_greedy_decoder is a special case of the ctc_beam_search_decoder with top_paths=1 (but that decoder is faster for this special case).
    # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=False)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=10, merge_repeated=False)
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    
    
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels), name="acc")

    init = tf.global_variables_initializer()

    def report_accuracy(decoded_list, test_labels):
        original_list = decode_sparse_tensor(test_labels)
        detected_list = decode_sparse_tensor(decoded_list)
        true_numer = 0
        all_numer = 0
        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
        print("T/F: original(length) <-------> detectcted(length)")
        for idx in range(min(len(original_list),len(detected_list))):
            number = original_list[idx]
            detect_number = detected_list[idx]  
            hit = (number == detect_number)          
            print(hit, list_to_chars(number), "(", len(number), ") <-------> ", list_to_chars(detect_number), "(", len(detect_number), ")")
            all_numer += len(number)
            for x in range(min(len(number),len(detect_number))):
                if number[x]==detect_number[x]:
                    true_numer += 1
        print("Test Accuracy:", true_numer * 1.0 / all_numer)

    def do_report():
        test_inputs,test_labels,test_seq_len = get_next_batch(TEST_BATCH_SIZE)
        test_feed = {inputs: test_inputs,
                     labels: test_labels,
                     seq_len: test_seq_len,
                     input_keep_prob: 1.0}
        dd = session.run(decoded[0], test_feed)
        report_accuracy(dd, test_labels)
 
    def do_batch():
        train_inputs, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)       
        feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len,
                input_keep_prob: 0.7, learning_rate: curr_learning_rate}        
        b_loss,b_labels, b_logits, b_seq_len,b_cost, steps, b_learning_rate, _ = session.run([loss, labels, logits, seq_len, cost, global_step, learning_rate, optimizer], feed)

        if steps > 0 and steps % REPORT_STEPS == 0:
            do_report()
        return b_cost, steps, b_learning_rate, train_inputs.shape[1]

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
            train_cost = 0
            for batch in range(BATCHES):
                start = time.time()
                c, steps, rate, width = do_batch()
                train_cost += c * BATCH_SIZE
                seconds = round(time.time() - start,2)
                print("step:", steps, "cost:", c, "batch seconds:", seconds, "learning rate:", rate, "width:", width)
                if np.isnan(c) or np.isinf(c):
                    print("Error: cost is nan or inf")
                    return                
                # if c < 100 and curr_learning_rate > 1e-6:
                #     curr_learning_rate = 1e-6           
                if c < 20 and curr_learning_rate > 1e-5:
                    curr_learning_rate = 1e-5
                if c < 1 and curr_learning_rate > 5e-5:
                    curr_learning_rate = 5e-5
                if c < 0.1 and curr_learning_rate > 1e-6:
                    curr_learning_rate = 1e-6

            # start = time.time()
            # train_inputs, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)
            # val_feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len, input_keep_prob: 1.0, learning_rate: curr_learning_rate }
            # val_cost, val_acc, steps = session.run([cost, acc, global_step], feed_dict=val_feed)
            # log = "steps: {}, val_cost: {:.3f}, val_acc: {:.3f}, time: {:.3f}s"
            # print(log.format(steps, val_cost, val_acc, time.time() - start))
            saver.save(session, checkpoint_path, global_step=steps)

if __name__ == '__main__':
    train()