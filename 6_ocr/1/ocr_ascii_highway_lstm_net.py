# coding=utf-8
# 中文OCR学习，尝试多层

import tensorflow as tf
import numpy as np
import os
import utils
import time
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import tensorflow.contrib.slim as slim
import math
import urllib,json,io

curr_dir = os.path.dirname(__file__)

image_height = 32

# LSTM
# num_hidden = 4
# num_layers = 1

# 所有 unicode CJK统一汉字（4E00-9FBB） + ascii的字符加 + ctc blank
# https://zh.wikipedia.org/wiki/Unicode
# https://zh.wikipedia.org/wiki/ASCII
ASCII_CHARS = [chr(c) for c in range(32,126+1)]
#ZH_CHARS = [chr(c) for c in range(int('4E00',16),int('9FBB',16)+1)]
#ZH_CHARS_PUN = ['。','？','！','，','、','；','：','「','」','『','』','‘','’','“','”',\
#                '（','）','〔','〕','【','】','—','…','–','．','《','》','〈','〉']

CHARS = ASCII_CHARS #+ ZH_CHARS + ZH_CHARS_PUN
# CHARS = ASCII_CHARS
num_classes = len(CHARS) + 1

#初始化学习速率
# LEARNING_RATE_INITIAL = 1e-3
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 500
MOMENTUM = 0.9

BATCHES = 64
BATCH_SIZE = 10
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))

# 增加 Highway 网络
def addHighwayLayer(inputs):
    H = slim.conv2d(inputs, 64, [3,3])
    T = slim.conv2d(inputs, 64, [3,3], biases_initializer = tf.constant_initializer(-1.0), activation_fn=tf.nn.sigmoid)    
    outputs = H * T + inputs * (1.0 - T)
    return outputs    

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, None, image_height], name="inputs")
    # 定义 ctc_loss 是稀疏矩阵
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    # 1维向量 size [batch_size] 等于 np.ones(batch_size)* image_width
    seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    drop_prob = 1 - keep_prob

    shape = tf.shape(inputs)
    batch_size, image_width = shape[0], shape[1]

    layer = tf.reshape(inputs, [batch_size,image_width,image_height,1])

    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm)
    for i in range(POOL_COUNT):
        for j in range(3):
            layer = addHighwayLayer(layer)
        layer = slim.conv2d(layer, 64, [3,3], stride=[2, 2], normalizer_fn=slim.batch_norm)  
        for j in range(3):
            layer = addHighwayLayer(layer)
   
    layer = tf.reshape(layer,[batch_size, -1, 64])  #[batch_size, image_width*image_height//POOL_SIZE//POOL_SIZE, 64]

    num_hidden = 128
    with tf.variable_scope('RNN1'):
        cell_fw = tf.contrib.rnn.GRUCell(num_hidden//2)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)    
        cell_bw = tf.contrib.rnn.GRUCell(num_hidden//2)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)    
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, layer, seq_len, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2) #[batch_size, image_width*image_height//POOL_SIZE//POOL_SIZE, num_hidden]
        layer = tf.reshape(outputs, [-1, num_hidden])    
    
    layer = tf.layers.dense(layer, num_hidden)
    layer = tf.reshape(layer, [batch_size, -1, num_hidden])    

    with tf.variable_scope('RNN2'):
        cell_fw = tf.contrib.rnn.GRUCell(num_hidden//2)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)    
        cell_bw = tf.contrib.rnn.GRUCell(num_hidden//2)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)    
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, layer, seq_len, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2) #[batch_size, image_width*image_height//POOL_SIZE//POOL_SIZE, num_hidden]
        layer = tf.reshape(outputs, [-1, num_hidden])
    # layer = tf.layers.dense(layer, 512, activation=tf.nn.relu)
    # layer = tf.layers.dropout(layer,drop_prob)
    
    # # 这里不需要再加上 tf.nn.softmax 层，因为ctc_loss会加
    layer = tf.layers.dense(layer, num_classes)

    # 输出对数： [batch_size , max_time , num_classes]
    layer = tf.reshape(layer, [batch_size, -1, num_classes])
    # 需要变换到 time_major == True [max_time x batch_size x num_classes]
    logits = tf.transpose(layer, (1, 0, 2), name="logits")

    return logits, inputs, labels, seq_len, keep_prob

def http(url,param=None):
    if param != None:
        paramurl = urllib.parse.urlencode(param)
        url = "%s?%s"%(url,paramurl)
    for i in range(3):
        try:
            r = urllib.request.urlopen(url, timeout=10)
            return r.read()
        except:
            pass
    raise Exception("can't open %s"%url)     

r = http('http://192.168.2.113:8888/')
fonts = json.loads(r.decode('utf-8'))
ENGFontNames = fonts['eng']
print("EngFontNames", ENGFontNames)
CHIFontNames = fonts['chi']
print("CHIFontNames", CHIFontNames)
AllFontNames = ENGFontNames + CHIFontNames

def getImage(CHARS, font_name, image_height, font_length, font_size, word_dict):
    text=''
    n = random.random()
    if n<0.1:
        for i in range(font_length):
            text += random.choice("123456789012345678901234567890-./$,:()+-*=><")
    elif n<0.5 and n>=0.1:
        for i in range(font_length):
            text += random.choice(CHARS)        
    else:
        while len(text)<font_length:
            word = random.choice(word_dict)
            _word=""
            for c in word:
                if c in CHARS:
                    _word += c
            text = text+" "+_word.strip()
    text = text.strip()

    params= {}
    params['text'] = text
    params['fontname'] = font_name
    params['fontsize'] = font_size
    #params['fontmode'] = random.choice([0,1,2,4,8])
    params['fontmode'] = random.choice([0,1,2,4])
    params['fonthint'] = random.choice([0,1,2,3,4,5])
    
    r = http('http://192.168.2.113:8888/',params)
    _img = Image.open(io.BytesIO(r))
    img=Image.new("RGB",_img.size,(255,255,255))
    img.paste(_img,(0,0),_img)
    img = utils.trim(img)
    w,h = img.size
    _h = random.randint(9,64)
    _w = round(w * _h / h)
    img = img.resize((_w,_h), Image.ANTIALIAS)
    img = np.asarray(img)
 #   img = utils.clearBackgroundColor(img)
    img = 1 - utils.img2gray(img)/255.   
    img = utils.dropZeroEdges(img)

    filter = np.random.random(img.shape) - 0.9
    filter = np.maximum(filter, 0) 
    img = img + filter * 5
    imin, imax = img.min(), img.max()
    img = (img - imin)/(imax - imin)

    img = utils.resize(img, image_height)
    return text, img

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 
# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch(batch_size=128):
    codes = []
    images = []   
    max_width_image = 0
    font_min_length = random.randint(10, 80)
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        font_length = random.randint(font_min_length-5, font_min_length+5)
        font_size = random.randint(9, 48)        
        text, image= getImage(CHARS, font_name, image_height, font_length, font_size, eng_world_list)
        images.append(image)
        if image.shape[1] > max_width_image: 
            max_width_image = image.shape[1]
        text_list = [CHARS.index(char) for char in text]
        codes.append(text_list)

    # 凑成4的整数倍
    max_width_image = max_width_image + (POOL_SIZE - max_width_image % POOL_SIZE)
    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = utils.img2vec(images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.transpose(image_vec)

    labels = [np.asarray(i) for i in codes]
    #labels转成稀疏矩阵
    sparse_labels = sparse_tuple_from(labels)
    #因为模型做了2次pool，所以 seq_len 也需要除以4
    seq_len = np.ones(batch_size) * (max_width_image * image_height) // (POOL_SIZE * POOL_SIZE)
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

    logits, inputs, labels, seq_len, keep_prob = neural_networks()

    # If time_major == True (default), this will be a Tensor shaped: [max_time x batch_size x num_classes]
    # 返回 A 1-D float Tensor, size [batch], containing the negative log probabilities.
    loss = tf.nn.ctc_loss(labels=labels,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss, name="cost")

    # 收敛效果不好
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=MOMENTUM).minimize(cost, global_step=global_step)

    # 做一个梯度裁剪，貌似也没啥用, 将梯度控制到 -1 和 1 之间
    # grads_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # grads_and_vars = grads_optimizer.compute_gradients(loss)
    # capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
    # gradients, variables = zip(*grads_optimizer.compute_gradients(loss))
    # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    # capped_grads_and_vars = zip(gradients, variables)

    #capped_grads_and_vars = [(tf.clip_by_norm(g, 5), v) for g,v in grads_and_vars]
    # optimizer = grads_optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)

    # 最小化 loss
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
        if len(original_list) != len(detected_list):
            print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                  " test and detect length desn't match")
        print("T/F: original(length) <-------> detectcted(length)")
        acc = 0.
        for idx in range(min(len(original_list),len(detected_list))):
            number = original_list[idx]
            detect_number = detected_list[idx]  
            hit = (number == detect_number)          
            print("%6s" % hit, list_to_chars(number), "(", len(number), ")")
            print("%6s" % "",  list_to_chars(detect_number), "(", len(detect_number), ")")
            # 计算莱文斯坦比
            import Levenshtein
            acc += Levenshtein.ratio(list_to_chars(number),list_to_chars(detect_number))
        print("Test Accuracy:", acc / len(original_list))

    def do_report():
        test_inputs,test_labels,test_seq_len = get_next_batch(TEST_BATCH_SIZE)
        test_feed = {inputs: test_inputs,
                     labels: test_labels,
                     seq_len: test_seq_len,
                     keep_prob: 1.0}
        dd = session.run(decoded[0], test_feed)
        report_accuracy(dd, test_labels)

    def restore(sess):
        curr_dir = os.path.dirname(__file__)
        model_dir = os.path.join(curr_dir, "model_ascii_highway_lstm")
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

                train_inputs, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)       
                feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len,
                        keep_prob: 0.95, learning_rate: curr_learning_rate}       

                # l=session.run(layer,feed)
                # print(train_inputs.shape)
                # print(l.shape)
                # print(train_seq_len[0])

                b_loss, b_labels, b_logits, b_seq_len, b_cost, steps, b_learning_rate, _ = \
                    session.run([loss, labels, logits, seq_len, cost, global_step, learning_rate, optimizer], feed)

                train_cost += b_cost * BATCH_SIZE
                seconds = round(time.time() - start,2)
                print("step:", steps, "cost:", b_cost, "batch seconds:", seconds, "learning rate:", b_learning_rate, "width:", train_seq_len[0])
                if np.isnan(b_cost) or np.isinf(b_cost):
                    print("Error: cost is nan or inf")
                    train_labels_list = decode_sparse_tensor(train_labels)
                    for i, train_label in enumerate(train_labels_list):
                        print(i,list_to_chars(train_label))
                    return   
                
                if seconds > 60: 
                    print('Exit for long time')
                    return

                if steps > 0 and steps % REPORT_STEPS == 0:
                    do_report()

            saver.save(session, checkpoint_path, global_step=steps)

if __name__ == '__main__':
    train()