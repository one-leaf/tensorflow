# coding=utf-8

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
import utils_pil, utils_font, utils_nn
import font_ascii_clean
import operator

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
LEARNING_RATE_INITIAL = 1e-4
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 1000
MOMENTUM = 0.9

BATCHES = 100
BATCH_SIZE = 2
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 4
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_ascii"

def RES(inputs, seq_len, reuse = False):
    with tf.variable_scope("OCR", reuse=reuse):
        print("inputs shape:",inputs.shape)
        layer = utils_nn.resNet101V2(inputs, True)    # N H/16 W/16 2048
        print("resNet shape:",layer.shape)

        shape = tf.shape(layer)
        batch_size, height, width, channel = shape[0], shape[1], shape[2], shape[3]
        
        layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu) 
        # N H W 1024
        layer = tf.reshape(layer, [batch_size*height, width, 1024]) # N*H, W, 1024
        print("resNet_seq shape:",layer.shape)
        layer.set_shape([None, None, 1024])
        layer = LSTM(layer, 512, 128)    # N*H, W, 512+128*2
        print("lstm shape:",layer.shape)
        layer = tf.reshape(layer, [batch_size, height, width, 512+128+128]) # N,H,W,512+128*2
        layer = slim.fully_connected(layer, 1024, normalizer_fn=None, activation_fn=None)  # N, H, W, 1024
        layer = tf.reshape(layer, [batch_size, height*width, 1024]) # N, W*H, 1024

        print("res fin shape:",layer.shape)
        return layer

def LSTM(inputs, fc_size, lstm_size):
    layer = inputs
    for i in range(4):
        with tf.variable_scope("rnn-%s"%i):
            layer = slim.fully_connected(layer, fc_size, normalizer_fn=slim.batch_norm, activation_fn=None)
            # layer = slim.fully_connected(layer, fc_size, normalizer_fn=None, activation_fn=None)
            cell_fw = tf.contrib.rnn.GRUCell(lstm_size, activation=tf.nn.relu)
            cell_bw = tf.contrib.rnn.GRUCell(lstm_size, activation=tf.nn.relu)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, layer, dtype=tf.float32)
            layer = tf.concat([outputs[0], outputs[1], layer], axis=-1)
            # layer = slim.batch_norm(layer)
    return layer

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, image_height, None, 1], name="inputs")
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    global_step = tf.Variable(0, trainable=False)

    net_res = RES(inputs, seq_len, reuse = False)
    res_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='OCR')

    # 需要变换到 time_major == True [max_time x batch_size x 2048]
    net_res = tf.transpose(net_res, (1, 0, 2))
    res_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=net_res, sequence_length=seq_len))
    res_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(res_loss, global_step=global_step, var_list=res_vars)
    res_decoded, _ = tf.nn.ctc_beam_search_decoder(net_res, seq_len, beam_width=10, merge_repeated=False)
    res_acc = tf.reduce_sum(tf.edit_distance(tf.cast(res_decoded[0], tf.int32), labels, normalize=False))
    res_acc = 1 - res_acc / tf.to_float(tf.size(labels.values))
    
    return  inputs, labels, global_step, \
            res_loss, res_optim, seq_len, res_acc, res_decoded
            

ENGFontNames, CHIFontNames = utils_font.get_font_names_from_url()
print("EngFontNames", ENGFontNames)
print("CHIFontNames", CHIFontNames)
AllFontNames = ENGFontNames + CHIFontNames
AllFontNames.remove("方正兰亭超细黑简体")
AllFontNames.remove("幼圆")
AllFontNames.remove("方正舒体")
AllFontNames.remove("方正姚体")
AllFontNames.remove("Impact")
AllFontNames.remove("Gabriola")

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 

def list_to_chars(list):
    try:
        return "".join([CHARS[v] for v in list])
    except Exception as err:
        return "Error: %s" % err        

if os.path.exists(os.path.join(curr_dir,"train.txt")):
    train_text_lines = open(os.path.join(curr_dir,"train.txt")).readlines()
else:
    train_text_lines = []

def get_next_batch_for_res(batch_size=128, _font_name=None, _font_size=None, _font_mode=None, _font_hint=None):
    inputs_images = []   
    codes = []
    max_width_image = 0
    info = []
    font_length = random.randint(5, 50)
    for i in range(batch_size):
        font_name = _font_name
        font_size = _font_size
        font_mode = _font_mode
        font_hint = _font_hint
        if font_name==None:
            font_name = random.choice(AllFontNames)
        if font_size==None:
            if random.random()>0.5:
                font_size = random.randint(9, 49)    
            else:
                font_size = random.randint(9, 15) 
        if font_mode==None:
            font_mode = random.choice([0,1,2,4]) 
        if font_hint==None:
            font_hint = random.choice([0,1,2,3,4,5])    
            # font_hint = random.choice([0,4]) 

        text  = utils_font.get_words_text(CHARS, eng_world_list, font_length)
        text = text + " " + "".join(random.sample(CHARS, random.randint(1,5)))
        text = text.strip()

            # random.shuffle(text)
            # text = "".join(text).strip()

        image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint )
        # image = utils_pil.convert_to_gray(image) 
        w, h = image.size
        if h > image_height:
            image = utils_pil.resize_by_height(image, image_height)  

        if random.random()>0.5:
            image = utils_pil.resize_by_height(image, image_height-random.randint(1,5))
            image, _ = utils_pil.random_space2(image, image,  image_height)
        
        # if if_to_G and random.random()>0.5:
        # if random.random()>0.5:
        #     image = utils_font.add_noise(image)   
        print(image.shape())
        image = np.asarray(image) 
        print(image.shape)

        image = utils.resize(image, height=image_height)

        if random.random()>0.5:
            image = image / 255.
        else:
            image = (255. - image) / 255.

        if max_width_image < image.shape[1]:
            max_width_image = image.shape[1]

        #image = image[:, : , np.newaxis]
        #image = np.reshape(image,(image.shape[0],image.shape[1],1))
        print(image.shape)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = image.eval()
        print(image.shape, type(image))

        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
        print(image.shape, type(image))
        image = image.eval()
        print(image.shape, type(image))
        inputs_images.append(image)
        codes.append([CHARS.index(char) for char in text])                  

        info.append([font_name, str(font_size), str(font_mode), str(font_hint)])

    # 凑成16的整数倍
    max_width_image = max_width_image + (POOL_SIZE - max_width_image % POOL_SIZE)

    inputs = np.zeros([batch_size, image_height, max_width_image, 1])
    for i in range(batch_size):
        image = utils.img2vec(inputs_images[i], height=image_height, width=max_width_image, flatten=False)
        # inputs[i,:] = image_vec # np.reshape(image_vec,(image_height, max_width_image, 1))
        # image = inputs_images[i], image_height, max_width_image)
        # image = image.eval()
        # print(image.shape)

        inputs[i,:] = image
        
    # print(inputs.shape)
    labels = [np.asarray(i) for i in codes]
    sparse_labels = utils.sparse_tuple_from(labels)
    seq_len = np.ones(batch_size) * (image_height*max_width_image)//(POOL_SIZE*POOL_SIZE)
    return inputs, sparse_labels, seq_len, info

def train():
    inputs, labels, global_step, \
        res_loss, res_optim, seq_len, res_acc, res_decoded = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_R_dir = os.path.join(model_dir, "RL32")

    if not os.path.exists(model_R_dir): os.mkdir(model_R_dir)
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        r_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='OCR'), sharded=True)

        ckpt = tf.train.get_checkpoint_state(model_R_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model OCR...")
            r_saver.restore(session, ckpt.model_checkpoint_path)    
            print("Restored.")

        AllLosts={}
        while True:
            errA = errD1 = errD2 = 1
            batch_size = BATCH_SIZE
            for batch in range(BATCHES):
                if len(AllLosts)>10 and random.random()>0.7:
                    sorted_font = sorted(AllLosts.items(), key=operator.itemgetter(1), reverse=True)
                    font_info = sorted_font[random.randint(0,10)]
                    font_info = font_info[0].split(",")
                    train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size, \
                        font_info[0], int(font_info[1]), int(font_info[2]), int(font_info[3]))
                else:
                    # train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size, False, _font_size=36)
                    train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size)
                # feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len} 
                start = time.time()                
                feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len} 

                errR, acc, _ , steps= session.run([res_loss, res_acc, res_optim, global_step], feed)
                font_info = train_info[0][0]+"/"+train_info[0][1]
                print("%d time: %4.4fs, res_acc: %.4f, res_loss: %.4f, info: %s " % (steps, time.time() - start, acc, errR, font_info))
                if np.isnan(errR) or np.isinf(errR) :
                    print("Error: cost is nan or inf")
                    return

                for info in train_info:
                    key = ",".join(info)
                    if key in AllLosts:
                        AllLosts[key]=AllLosts[key]*0.95+errR*0.05
                    else:
                        AllLosts[key]=errR

                # 报告
                if steps >0 and steps % REPORT_STEPS < 2:
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
                    sorted_fonts = sorted(AllLosts.items(), key=operator.itemgetter(1), reverse=True)
                    for f in sorted_fonts[:20]:
                        print(f)

            print("Save Model OCR ...")
            r_saver.save(session, os.path.join(model_R_dir, "OCR.ckpt"), global_step=steps)         

if __name__ == '__main__':
    train()