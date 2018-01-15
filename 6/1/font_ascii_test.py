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
image_size = 512

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
MOMENTUM = 0.9

BATCHES = 256
BATCH_SIZE = 4
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_ascii_srgan"
SEQ_LENGHT = (image_size * image_size ) // (POOL_SIZE * POOL_SIZE)

def TRIM_G(inputs, reuse=False):    
    with tf.variable_scope("TRIM_G", reuse=reuse):      
        layer, half_layer = utils_nn.pix2pix_g2(inputs)
        # print(half_layer.shape) ? 1,1, 512
        return layer, half_layer

def RES(inputs, keep_prob, seq_len, reuse = False):
    with tf.variable_scope("OCR", reuse=reuse):
        batch_size = tf.shape(inputs)[0]
        layer = utils_nn.resNet50(inputs, True)
        # layer = slim.fully_connected(layer, 1024, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        # layer = slim.dropout(layer, keep_prob)

        # layer = tf.reshape(layer, [batch_size, SEQ_LENGHT, 1024])
        # layer = LSTM(layer, keep_prob, seq_len)
        # layer = tf.reshape(lstm_layer, [batch_size, -1, 1024])

        # layer = lstm_layer
        # layer = tf.concat([layer, lstm_layer], axis=2)

        layer = slim.fully_connected(layer, 1024, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)        
        # layer = slim.dropout(layer, keep_prob)
        layer = slim.fully_connected(layer, CLASSES_NUMBER, normalizer_fn=None, activation_fn=None)  

        layer = tf.reshape(layer, [batch_size, -1, CLASSES_NUMBER])       
        return layer

# 输入 half_layer
def LSTM(inputs, keep_prob, seq_len):
    # layer = slim.fully_connected(inputs, SEQ_LENGHT, normalizer_fn=None, activation_fn=None)
    # layer = tf.reshape(inputs, (-1, SEQ_LENGHT, POOL_SIZE*POOL_SIZE))
    num_hidden = 256
    cell_fw = tf.contrib.rnn.GRUCell(num_hidden//2)
    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)    
    cell_bw = tf.contrib.rnn.GRUCell(num_hidden//2)
    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob, output_keep_prob=keep_prob)    
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, seq_len, dtype=tf.float32)
    layer = tf.concat(outputs, axis=2)
    # layer = slim.fully_connected(layer, 1024, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu) 
    # layer = slim.dropout(layer, keep_prob)
    return layer

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size], name="inputs")
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, trainable=False)

    layer = tf.reshape(inputs, (-1, image_size, image_size, 1))
    # resize_layer = tf.image.resize_images(layer, (image_size//2,image_size//2), method=tf.image.ResizeMethod.BILINEAR)
    # print(resize_layer.shape)

    net_g, half_net_g = TRIM_G(layer, reuse = False)

    net_res = RES(layer, keep_prob, seq_len, reuse = False)
    res_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='OCR')
    # 需要变换到 time_major == True [max_time x batch_size x 2048]
    net_res = tf.transpose(net_res, (1, 0, 2))
    res_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=net_res, sequence_length=seq_len))
    res_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(res_loss, global_step=global_step, var_list=res_vars)
    res_decoded, _ = tf.nn.ctc_beam_search_decoder(net_res, seq_len, beam_width=10, merge_repeated=False)
    res_acc = tf.reduce_sum(tf.edit_distance(tf.cast(res_decoded[0], tf.int32), labels, normalize=False))
    res_acc = 1 - res_acc / tf.to_float(tf.size(labels.values))

    
    return  inputs, labels, global_step, keep_prob, \
            res_loss, res_optim, seq_len, res_acc, res_decoded, \
            net_g

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
    return "".join([CHARS[v] for v in list])

def get_next_batch_for_res(batch_size=128, if_to_G=True, _font_name=None, _font_size=None, _font_mode=None, _font_hint=None):
    inputs_images = []   
    codes = []
    max_width_image = 0
    info = []
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

        while True:
            font_length = random.randint(5, 400)
            text  = utils_font.get_words_text(CHARS, eng_world_list, font_length)
            image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint )
            temp_image = utils_pil.resize_by_height(image, image_height)
            w, h = temp_image.size            
            if w * h < image_size * image_size: break

        image = utils_pil.convert_to_gray(image) 
        w, h = image.size
        image = utils_pil.resize_by_height(image, image_height)  

        # if if_to_G and random.random()>0.5:
        #     _h =  random.randint(9, image_height+1)
        #     image = utils_pil.resize_by_height(image, _h) 

        # if if_to_G:
        #     image = utils_pil.random_space2(image, image_height)
        #     image = utils_font.add_noise(image)   
    
        # image = np.asarray(image) 

        # if not if_to_G:    
        #     image = utils.resize(image, height=image_height)
        #     image = utils.img2bw(image)

        # if if_to_G:
        #     image = image * random.uniform(0.3, 1)        

        # if if_to_G and random.random()>0.5:
        #     image = image / 255.
        # else:
        #     image = (255. - image) / 255.

        image = np.asarray(image)
        image = (255. - image) / 255.
        inputs_images.append(image)
        codes.append([CHARS.index(char) for char in text])                  

        info.append([font_name, str(font_size), str(font_mode), str(font_hint)])

    inputs = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        inputs[i,:] = utils.square_img(inputs_images[i],np.zeros([image_size, image_size]))

    labels = [np.asarray(i) for i in codes]
    sparse_labels = utils.sparse_tuple_from(labels)
    seq_len = np.ones(batch_size) * SEQ_LENGHT                
    return inputs, sparse_labels, seq_len, info

def train():
    inputs, labels, global_step, keep_prob,\
        res_loss, res_optim, seq_len, res_acc, res_decoded, \
        net_g = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_G_dir = os.path.join(model_dir, "TG")
    model_R_dir = os.path.join(model_dir, "OCR")

    if not os.path.exists(model_R_dir): os.mkdir(model_R_dir)
    if not os.path.exists(model_G_dir): os.mkdir(model_G_dir)  
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        r_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='OCR'), sharded=True)
        g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TRIM_G'), sharded=False)

        # ckpt = tf.train.get_checkpoint_state(model_G_dir)
        # if ckpt and ckpt.model_checkpoint_path:           
        #     print("Restore Model G...")
        #     g_saver.restore(session, ckpt.model_checkpoint_path)   

        ckpt = tf.train.get_checkpoint_state(model_R_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model OCR...")
            r_saver.restore(session, ckpt.model_checkpoint_path)    

        AllLosts={}
        while True:
            errA = errD1 = errD2 = 1
            batch_size = 1
            for batch in range(BATCHES):
                if len(AllLosts)>10 and random.random()>0.7:
                    sorted_font = sorted(AllLosts.items(), key=operator.itemgetter(1), reverse=True)
                    font_info = sorted_font[random.randint(0,10)]
                    font_info = font_info[0].split(",")
                    train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size, False, \
                        font_info[0], int(font_info[1]), int(font_info[2]), int(font_info[3]))
                else:
                    # train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size, False, _font_size=36)
                    train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size)
                # feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len} 
                start = time.time() 

                # p_net_g = session.run(net_g, {inputs: train_inputs}) 

                # p_net_g = np.squeeze(p_net_g, axis=3)
                # for i in range(batch_size):
                #     _t_img = utils.unsquare_img(p_net_g[i], image_height)                        
                #     _t_img = utils.cvTrimImage(_t_img)
                #     _t_img[_t_img<0] = 0
                #     _t_img = utils.resize(_t_img, image_height)
                #     if _t_img.shape[0] * _t_img.shape[1] <= image_size * image_size:
                #         p_net_g[i] = utils.square_img(_t_img, np.zeros([image_size, image_size]), image_height)

                # feed = {inputs: p_net_g, labels: train_labels, seq_len: train_seq_len, keep_prob: 0.95} 
                feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len, keep_prob: 0.95} 

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
                if steps == 1 or steps % REPORT_STEPS == 0:
                    train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(batch_size)   
                    # p_net_g = session.run(resize_layer, {inputs: train_inputs}) 
                    # p_net_g = np.squeeze(p_net_g, axis=3)

                    # for i in range(batch_size):
                    #     _t_img = utils.unsquare_img(p_net_g[i], image_height)                        
                    #     _t_img_bin = np.copy(_t_img)    
                    #     _t_img_bin[_t_img_bin<=0.3] = 0
                    #     _t_img = utils.dropZeroEdges(_t_img_bin, _t_img, min_rate=0.1)
                    #     _t_img = utils.resize(_t_img, image_height)
                    #     if _t_img.shape[0] * _t_img.shape[1] <= image_size * image_size:
                    #         p_net_g[i] = utils.square_img(_t_img, np.zeros([image_size, image_size]), image_height)

                    decoded_list = session.run(res_decoded[0], {inputs: train_inputs, seq_len: train_seq_len, keep_prob: 1}) 

                    for i in range(batch_size): 
                        # _img = np.vstack((train_inputs[i], p_net_g[i])) 
                        cv2.imwrite(os.path.join(curr_dir,"test","%s_%s.png"%(steps,i)), train_inputs[i] * 255) 
                        # cv2.imwrite(os.path.join(curr_dir,"test","%s_%s.png"%(steps,i)), p_net_g[i] * 255) 

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
                        print("%6s" % hit, list_to_chars(number), "(", len(number), ")")
                        print("%6s" % "",  list_to_chars(detect_number), "(", len(detect_number), ")")
                        # 计算莱文斯坦比
                        import Levenshtein
                        acc += Levenshtein.ratio(list_to_chars(number),list_to_chars(detect_number))
                    print("Test Accuracy:", acc / len(original_list))
                    sorted_fonts = sorted(AllLosts.items(), key=operator.itemgetter(1), reverse=True)
                    for f in sorted_fonts[:20]:
                        print(f)
            print("Save Model OCR ...")
            r_saver.save(session, os.path.join(model_R_dir, "OCR.ckpt"), global_step=steps)
            # try:
            #     ckpt = tf.train.get_checkpoint_state(model_G_dir)
            #     if ckpt and ckpt.model_checkpoint_path:           
            #         print("Restore Model G...")
            #         g_saver.restore(session, ckpt.model_checkpoint_path)   
            # except:
            #     pass

if __name__ == '__main__':
    train()