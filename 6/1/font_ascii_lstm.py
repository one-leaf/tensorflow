# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import utils
import time
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
from tensorflow.contrib import slim 
from tensorflow.contrib.slim.nets import inception, resnet_v2
import math
import urllib,json,io
import utils_pil, utils_font, utils_nn
import operator
from collections import deque

curr_dir = os.path.dirname(__file__)

image_height = 32
# image_size = 512
# resize_image_size = 256
# 所有 unicode CJK统一汉字（4E00-9FBB） + ascii的字符加 + ctc blank
# https://zh.wikipedia.org/wiki/Unicode
# https://zh.wikipedia.org/wiki/ASCII
ASCII_CHARS = [chr(c) for c in range(32,126+1)]
#ZH_CHARS = [chr(c) for c in range(int('4E00',16),int('9FBB',16)+1)]
#ZH_CHARS_PUN = ['。','？','！','，','、','；','：','「','」','『','』','‘','’','“','”',\
#                '（','）','〔','〕','【','】','—','…','–','．','《','》','〈','〉']

CHARS = ASCII_CHARS #+ ZH_CHARS + ZH_CHARS_PUN
# CHARS = ASCII_CHARS
CLASSES_NUMBER = len(CHARS) + 1 

#初始化学习速率
LEARNING_RATE_INITIAL = 1e-3
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 500

BATCHES = 50
BATCH_SIZE = 1
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 4
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_ascii_cnn_lstm"
MAX_IMAGE_WIDTH = 4096
SEQ_LENGTH  = 255

def RES(inputs, seq_len, reuse = False):
    with tf.variable_scope("OCR", reuse=reuse):
        print("inputs shape:",inputs.shape)
        # layer = utils_nn.resNet101V2(inputs, True)    # N H W/16 2048
        # layer = utils_nn.resNet50(inputs, True, [2,1]) # (N H/16 W 2048)

        # with slim.arg_scope(inception.inception_v3_arg_scope()):
        #     with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=True):
        #         layer, _ = inception.inception_v3_base(inputs, final_endpoint="Mixed_5d")

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            layer, _ = resnet_v2.resnet_v2_152(inputs,
                                                None,
                                                is_training=True,
                                                global_pool=False,
                                                output_stride=16) 
        print("ResNet shape:",layer.shape)

        # 直接将网络拉到256 [N 1 256 256]
        with tf.variable_scope("Normalize"):
            layer = slim.conv2d(layer, 1024, [2,2], [2,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
            layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
            layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 

        # layer = utils_nn.resNet101(inputs, True)
                    
        # with tf.variable_scope("ResNext"):
        #     layer = slim.conv2d(inputs, 64, [2,4], [2,4], normalizer_fn=slim.batch_norm, activation_fn=None) 
        #     tf.summary.image('1_2_4_zoom', tf.transpose (layer, [3, 1, 2, 0]), max_outputs=6)
        #     layer = utils_nn.resNext50(layer, True, [2,1]) # (N H/16 W 2048)
        #     tf.summary.image('2_res50', tf.transpose (layer, [3, 1, 2, 0]), max_outputs=6)

        temp_layer = layer
        # with tf.variable_scope("Normalize"):
        #     layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        #     layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
        #     layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
            # layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None) 
       
        # 将图像高度和宽度 // [2, 4]
        # layer = slim.avg_pool2d(layer, [2, 4], [2, 4]) 
        print("ResNet shape:",layer.shape)

        # 增加坐标信息，增加的个数为 embedd_size
        # max_width_height, embedd_size
        # max_width_height 为缩放后的 w 的最大宽度，实际上的最大图片宽度为 max_width_height * 4
        with tf.variable_scope("Coordinates"):
            max_width_height = MAX_IMAGE_WIDTH//16
            embedd_size = 64
            layer = Coordinates(layer, max_width_height, embedd_size)
            print("Coordinates shape:",layer.shape)

        with tf.variable_scope("LSTM"):
            layer = tf.squeeze(layer, squeeze_dims=1)
            print("SEQ shape:",layer.shape)
            layer = LSTM(layer, 256+embedd_size, seq_len)    # N, W*H, 256
            print("lstm shape:",layer.shape)

        return layer, temp_layer

# 插入像素的坐标信息
def Coordinates(inputs, max_width_height, embedd_size):
    shape = tf.shape(inputs)
    batch_size, h, w = shape[0],shape[1],shape[2]
    x = tf.range(w*h)
    x = tf.reshape(x, [1, h, w, 1])
    loc = tf.tile(x, [batch_size, 1, 1, 1])
    embedding = tf.get_variable("embedding", initializer=tf.random_uniform([max_width_height, embedd_size], -1.0, 1.0)) 
    loc = tf.nn.embedding_lookup(embedding, loc)
    loc = tf.squeeze(loc, squeeze_dims=3)
    loc = tf.concat([inputs, loc], 3)
    return loc

# 采用标准正交基的方式初始化参数
def orthogonal_initializer(shape, dtype=tf.float32, *args, **kwargs):
    del args
    del kwargs
    # 扁平化shape
    flat_shape = (shape[0], np.prod(shape[1:]))
    # 标准正态分布
    w = np.random.randn(*flat_shape)
    # 奇异值分解，提取矩阵特征，返回的 u,v 都是正交随机的，
    # u,v 分别代表了batch之间的数据相关性和同一批数据之间的相关性，_ 值代表了批次和数据的交叉相关性，舍弃掉
    u, _, v = np.linalg.svd(w, full_matrices=False)
    w = u if u.shape == flat_shape else v
    return tf.constant(w.reshape(shape), dtype=dtype)

# LSTM 不能加上 batch_norm，会抹杀特征，在ctc中很难学习到东西
# 如果用relu代替默认的tanh会收敛快很多，但后期网络很难收敛
def LSTM(inputs, lstm_size, seq_len):
    layer = inputs
    for i in range(3):
        with tf.variable_scope("rnn-%s"%i):
            cell_fw = tf.contrib.rnn.GRUCell(lstm_size, 
                kernel_initializer=orthogonal_initializer,
                bias_initializer=tf.zeros_initializer)
            cell_bw = tf.contrib.rnn.GRUCell(lstm_size, 
                kernel_initializer=orthogonal_initializer,
                bias_initializer=tf.zeros_initializer)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, layer, sequence_length=seq_len, dtype=tf.float32)
        layer += tf.nn.leaky_relu(outputs[0]+outputs[1])
    return layer

# 这个模型中后期的网络非常不稳定
# def LSTM(inputs, lstm_size, seq_len):
#     layer = inputs
#     for i in range(2):
#         with tf.variable_scope("rnn-%s"%i):
#             cell_fw = tf.contrib.rnn.GRUCell(lstm_size, 
#                 kernel_initializer=orthogonal_initializer,
#                 bias_initializer=tf.zeros_initializer)
#             cell_bw = tf.contrib.rnn.GRUCell(lstm_size, 
#                 kernel_initializer=orthogonal_initializer,
#                 bias_initializer=tf.zeros_initializer)
#             outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, layer, sequence_length=seq_len, dtype=tf.float32)
#         net = tf.concat(outputs, -1)  
#         net = slim.fully_connected(net, lstm_size, normalizer_fn=None, activation_fn=None)
#         layer = tf.nn.leaky_relu(net + layer)
#     return layer

# def LSTM(inputs, lstm_size, seq_len):
#     layer = inputs
#     cells_fw = [tf.contrib.rnn.GRUCell(lstm_size//2, 
#                 kernel_initializer=orthogonal_initializer,
#                 bias_initializer=tf.zeros_initializer) for _ in range(3)]
#     cells_bw = [tf.contrib.rnn.GRUCell(lstm_size//2, 
#                 kernel_initializer=orthogonal_initializer,
#                 bias_initializer=tf.zeros_initializer) for _ in range(3)]
#     outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, layer, sequence_length=seq_len, dtype=tf.float32)
#     net = tf.concat(outputs, -1)  
#     net = slim.fully_connected(net, lstm_size, normalizer_fn=None, activation_fn=None)
#     layer = tf.nn.leaky_relu(net + layer)
#     return layer

# def LSTM(inputs, lstm_size, seq_len):
#     layer = inputs
#     convolved = tf.transpose(layer, [1, 0, 2])
#     lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
#         num_layers=4,
#         num_units=lstm_size,
#         dropout=0,
#         dtype=tf.float32,
#         kernel_initializer=orthogonal_initializer,
#         bias_initializer=tf.zeros_initializer(),
#         direction="bidirectional")
#     outputs, _ = lstm(convolved)
#     net = tf.transpose(outputs, [1, 0, 2])
#     net = slim.fully_connected(net, lstm_size, normalizer_fn=None, activation_fn=tf.nn.leaky_relu)
#     return layer


# 失败的模型，最后会导致所有的softmax都为1 
# def LSTM(inputs, lstm_size, seq_len):
#     layer = inputs
#     convolved = tf.transpose(layer, [1, 0, 2])
#     lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
#         num_layers=4,
#         num_units=lstm_size,
#         dropout=0,
#         dtype=tf.float32,
#         kernel_initializer=orthogonal_initializer,
#         bias_initializer=tf.zeros_initializer(),
#         direction="bidirectional")
#     outputs, _ = lstm(convolved)
#     outputs = tf.transpose(outputs, [1, 0, 2])
#     net = slim.fully_connected(outputs, 2, normalizer_fn=None, activation_fn=None)
#     net =  tf.nn.softmax(net)
#     masks = tf.expand_dims(tf.gather(net, axis = -1, indices = 0), -1)
#     # 这种直接拉到0，有点太粗暴
#     # masks = tf.expand_dims(tf.cast(tf.argmax(net, -1), tf.float32),-1)
#     tf.summary.image('masks', tf.expand_dims(masks,-1), max_outputs=9)
#     layer = layer * masks 
#     return layer

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, image_height, None, 1], name="inputs")
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    global_step = tf.Variable(0, trainable=False)
    lr = tf.Variable(LEARNING_RATE_INITIAL, trainable=False)

    net_res, temp_layer= RES(inputs, seq_len, reuse = False)
    res_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='OCR')

    # 需要变换到 time_major == True [max_time x batch_size x 2048]
    net_res = tf.transpose(net_res, (1, 0, 2))
    res_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=net_res, sequence_length=seq_len))
    # res_optim = tf.train.AdamOptimizer(lr).minimize(res_loss, global_step=global_step, var_list=res_vars)
    res_optim = tf.train.AdamOptimizer(lr).minimize(res_loss, global_step=global_step)
 
    # 防止梯度爆炸
    # res_optim = tf.train.AdamOptimizer(lr)
    # gvs = res_optim.compute_gradients(res_loss)
    # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    # res_optim = res_optim.apply_gradients(capped_gvs, global_step=global_step)

    res_decoded, _ = tf.nn.ctc_beam_search_decoder(net_res, seq_len, beam_width=10, merge_repeated=False)
    res_acc = tf.reduce_sum(tf.edit_distance(tf.cast(res_decoded[0], tf.int32), labels, normalize=False))
    res_acc = 1 - res_acc / tf.to_float(tf.size(labels.values))


    # 加入日志
    tf.summary.scalar('res_loss', res_loss)
    tf.summary.scalar('res_acc', res_acc)
    # res_images = res_layer[-1]
    # res_images = tf.transpose(res_images, perm=[2, 0, 1])
    # tf.summary.image('net_res', tf.expand_dims(res_images,-1), max_outputs=9)
    for var in res_vars:
        tf.summary.histogram(var.name, var)
    summary = tf.summary.merge_all()

    return  inputs, labels, global_step, lr, summary, \
            res_loss, res_optim, seq_len, res_acc, res_decoded, temp_layer

def list_to_chars(list):
    try:
        return "".join([CHARS[v] for v in list])
    except Exception as err:
        return "Error: %s" % err        

def dataset_init():
    data_dir = os.path.join(curr_dir,"data")
    datafiles = os.listdir(data_dir)
    data_file = os.path.join(data_dir, random.choice(datafiles))
    print("load data_file", data_file)
    return tf.python_io.tf_record_iterator(data_file)

dataset = dataset_init()
dataset_example=tf.train.Example() 

def get_next_batch_for_res(batch_size=128):
    inputs_images = []   
    codes = []
    max_width_image = 0
    info = []
    seq_len = np.ones(batch_size)

    for i in range(batch_size):
        serialized_example = next(dataset, None)
        if serialized_example==None:
            raise Exception("has finished train one data file, stop")

        dataset_example.ParseFromString(serialized_example)

        font_name = str(dataset_example.features.feature['font_name'].bytes_list.value[0],  encoding="utf-8")
        font_size = dataset_example.features.feature['font_size'].int64_list.value[0]
        font_mode = dataset_example.features.feature['font_mode'].int64_list.value[0]
        font_hint = dataset_example.features.feature['font_mode'].int64_list.value[0]

        text = str(dataset_example.features.feature['label'].bytes_list.value[0],  encoding="utf-8")
        size = dataset_example.features.feature['size'].int64_list.value
        image = dataset_example.features.feature['image'].bytes_list.value[0]
        image = utils_pil.frombytes(tuple(size), image)

        image = utils_pil.convert_to_gray(image) 
        w, h = size
        if h > image_height:
            image = utils_pil.resize_by_height(image, image_height)  

        image = utils_pil.resize_by_height(image, image_height-random.randint(1,5))
        image, _ = utils_pil.random_space2(image, image,  image_height)
        
        image = utils_font.add_noise(image)   
        image = np.asarray(image) 

        image = utils.resize(image, image_height, MAX_IMAGE_WIDTH)

        if random.random()>0.5:
            image = image / 255.
        else:
            image = (255. - image) / 255.

        if max_width_image < image.shape[1]:
            max_width_image = image.shape[1]
          
        inputs_images.append(image)
        codes.append([CHARS.index(char) for char in text])                  

        info.append([font_name, str(font_size), str(font_mode), str(font_hint), str(len(text))])
        seq_len[i]=len(text)+1

    # 凑成4的整数倍
    # if max_width_image % 4 > 0:
    #     max_width_image = max_width_image + 4 - max_width_image % 4

    # 如果图片超过最大宽度
    if max_width_image < MAX_IMAGE_WIDTH:
        max_width_image = MAX_IMAGE_WIDTH
        # raise Exception("img width must %s <= %s " % (max_width_image, MAX_IMAGE_WIDTH))

    inputs = np.zeros([batch_size, image_height, max_width_image, 1])
    for i in range(batch_size):
        image_vec = utils.img2vec(inputs_images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.reshape(image_vec,(image_height, max_width_image, 1))
     
    # print(inputs.shape, len(codes))
    labels = [np.asarray(i) for i in codes]
    sparse_labels = utils.sparse_tuple_from(labels)

    # max_width_image = math.ceil((max_width_image-3+1.)/2.)
    # max_width_image = math.ceil((max_width_image-3+1.)/1.)
    # max_width_image = math.ceil((max_width_image-3+1.)/2.)
    # max_width_image = math.ceil((max_width_image-3+1.)/1.)
    # max_width_image = math.ceil((max_width_image-3+1.)/2.)

    seq_len = np.ones(batch_size) * SEQ_LENGTH
    # print(inputs.shape, seq_len.shape, [len(l) for l in labels])
    return inputs, sparse_labels, seq_len, info

# get_next_batch_for_res(1)

def train():
    inputs, labels, global_step, lr, summary, \
        res_loss, res_optim, seq_len, res_acc, res_decoded, net_res = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_R_dir = os.path.join(model_dir, "RL32")
    if not os.path.exists(model_R_dir): os.mkdir(model_R_dir)

    log_dir = os.path.join(model_dir, "logs")
    if not os.path.exists(log_dir): os.mkdir(log_dir)

    with tf.Session() as session:
        print("tf init")
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # session.run(init_op)
        session.run(tf.global_variables_initializer())

        print("tf check restore")
        # r_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='OCR'), sharded=True, max_to_keep=5)
        r_saver = tf.train.Saver(max_to_keep=5)

        for i in range(3):
            ckpt = tf.train.get_checkpoint_state(model_R_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Restore Model OCR...")
                stem = os.path.basename(ckpt.model_checkpoint_path)
                restore_iter = int(stem.split('-')[-1])
                try:
                    r_saver.restore(session, ckpt.model_checkpoint_path)    
                except:
                    new_restore_iter = restore_iter - BATCHES
                    with open(os.path.join(model_R_dir,"checkpoint"),'w') as f:
                        f.write('model_checkpoint_path: "OCR.ckpt-%s"\n'%new_restore_iter)
                        f.write('all_model_checkpoint_paths: "OCR.ckpt-%s"\n'%new_restore_iter)
                    continue
                session.run(tf.assign(global_step, restore_iter))
                if restore_iter<10000:        
                    session.run(tf.assign(lr, 1e-4))
                elif restore_iter<50000:            
                    session.run(tf.assign(lr, 1e-5))
                else:
                    session.run(tf.assign(lr, 1e-6))
                print("Restored to %s."%restore_iter)
                break
            else:
                break
            print("restored fail, return")
            return

        print("tf create summary")
        train_writer = tf.summary.FileWriter(log_dir, session.graph)

        print("tf train")

        AllLosts={}
        accs = deque(maxlen=200)
        losts = deque(maxlen=200)
        while True:
            errR = 1
            batch_size = BATCH_SIZE
            for batch in range(BATCHES):
                start = time.time()    
                train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size)
                feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len} 

                feed_time = time.time() - start
                start = time.time()    

                # _res = session.run(net_res, feed)
                # print(train_inputs.shape)
                # print(_res.shape)
                # print(train_seq_len[0])

                errR, acc, _ , steps, res_lr = session.run([res_loss, res_acc, res_optim, global_step, lr], feed)
                font_length = int(train_info[0][-1])
                font_info = train_info[0][0]+"/"+train_info[0][1]+"/"+str(font_length)
                accs.append(acc)
                avg_acc = sum(accs)/len(accs)

                losts.append(errR)
                avg_losts = sum(losts)/len(losts)

                # errR = errR / font_length
                print("%s, %d time: %4.4fs / %4.4fs, acc: %.4f / %.4f, loss: %.4f / %.4f, lr:%.8f, info: %s " % \
                    (time.ctime(), steps, feed_time, time.time() - start, acc, avg_acc, errR, avg_losts, res_lr, font_info))

                # 如果当前lost低于平均lost，就多训练
                need_reset_global_step = False
                for _ in range(10):
                    if errR <=  avg_losts*2: break 
                    start = time.time()                
                    errR, acc, _, res_lr = session.run([res_loss, res_acc, res_optim, lr], feed)
                    accs.append(acc)
                    avg_acc = sum(accs)/len(accs)                  
                    print("%s, %d time: 0.0000s / %4.4fs, acc: %.4f, avg_acc: %.4f, loss: %.4f, avg_loss: %.4f, lr:%.8f, info: %s " % \
                        (time.ctime(), steps, time.time() - start, acc, avg_acc, errR, avg_losts, res_lr, font_info))   
                    need_reset_global_step = True
                    
                if need_reset_global_step:                     
                    session.run(tf.assign(global_step, steps))

                # if np.isnan(errR) or np.isinf(errR) :
                #     print("Error: cost is nan or inf")
                #     return

                for info in train_info:
                    key = ",".join(info)
                    if key in AllLosts:
                        AllLosts[key]=AllLosts[key]*0.99+acc*0.01
                    else:
                        AllLosts[key]=acc

                if acc/avg_acc<=0.2:
                    for i in range(batch_size): 
                        filename = "%s_%s_%s_%s_%s_%s_%s.png"%(acc, steps, i, \
                            train_info[i][0], train_info[i][1], train_info[i][2], train_info[i][3])
                        cv2.imwrite(os.path.join(curr_dir,"test",filename), train_inputs[i] * 255)                    
                # 报告
                if steps >0 and steps % REPORT_STEPS == 0:
                    train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size)   
           
                    decoded_list = session.run(res_decoded[0], {inputs: train_inputs, seq_len: train_seq_len}) 

                    for i in range(batch_size): 
                        cv2.imwrite(os.path.join(curr_dir,"test","%s_%s.png"%(steps,i)), train_inputs[i] * 255) 

                    original_list = utils.decode_sparse_tensor(train_labels)
                    detected_list = utils.decode_sparse_tensor(decoded_list)
                    if len(original_list) != len(detected_list):
                        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                            " test and detect length desn't match")
                    print("T/F: original(length) <-------> detectcted(length)")
                    acc = 0.
                    for idx in range(min(len(original_list),len(detected_list))):
                        number = original_list[idx]
                        detect_number = detected_list[idx]  
                        hit = (number == detect_number)
                        print("----------",hit,"------------")          
                        print(list_to_chars(number), "(", len(number), ")")
                        print(list_to_chars(detect_number), "(", len(detect_number), ")")
                        # 计算莱文斯坦比
                        import Levenshtein
                        acc += Levenshtein.ratio(list_to_chars(number),list_to_chars(detect_number))
                    print("Test Accuracy:", acc / len(original_list))
                    sorted_fonts = sorted(AllLosts.items(), key=operator.itemgetter(1), reverse=False)
                    for f in sorted_fonts[:20]:
                        print(f)

                    if avg_losts>100:        
                        session.run(tf.assign(lr, 1e-4))
                    elif avg_losts>10:            
                        session.run(tf.assign(lr, 5e-5))
                    else:
                        session.run(tf.assign(lr, 1e-5))
                                            
            # 如果当前 loss 为 nan，就先不要保存这个模型
            if np.isnan(errR) or np.isinf(errR):
                continue
            print("Save Model OCR ...")
            r_saver.save(session, os.path.join(model_R_dir, "OCR.ckpt"), global_step=steps)         
            logs = session.run(summary, feed)
            train_writer.add_summary(logs, steps)

if __name__ == '__main__':
    train()