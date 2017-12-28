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

curr_dir = os.path.dirname(__file__)

image_height = 32
image_size = 256

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
LEARNING_RATE_INITIAL = 2e-4
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

def SRGAN_g(inputs, reuse=False):    
    with tf.variable_scope("SRGAN_g", reuse=reuse):      
        layer, half_layer = utils_nn.pix2pix_g2(inputs)
        return layer, half_layer

def RES(inputs, reuse = False):
    with tf.variable_scope("RES", reuse=reuse):
        layer, conv = utils_nn.resNet50(inputs, True)
        shape = tf.shape(inputs)
        batch_size = shape[0] 
        layer = slim.fully_connected(layer, 1000, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer = slim.fully_connected(layer, CLASSES_NUMBER, normalizer_fn=None, activation_fn=None)  
        layer = tf.reshape(layer, [batch_size, -1, CLASSES_NUMBER])
        return layer

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size], name="inputs")
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    global_step = tf.Variable(0, trainable=False)

    layer = tf.reshape(inputs, (-1, image_size, image_size, 1))

    net_res = RES(layer, reuse = False)
    seq_len = tf.placeholder(tf.int32, [None], name="seq_len")
    res_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RES')
    # 需要变换到 time_major == True [max_time x batch_size x 2048]
    net_res = tf.transpose(net_res, (1, 0, 2))
    res_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=net_res, sequence_length=seq_len))
    res_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(res_loss, global_step=global_step, var_list=res_vars)
    res_decoded, _ = tf.nn.ctc_beam_search_decoder(net_res, seq_len, beam_width=10, merge_repeated=False)
    res_acc = tf.reduce_sum(tf.edit_distance(tf.cast(res_decoded[0], tf.int32), labels, normalize=False))
    res_acc = 1 - res_acc / tf.to_float(tf.size(labels.values))

    net_g = SRGAN_g(layer, reuse = False)
    g_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g')
    
    return  inputs, labels, global_step, \
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

def get_next_batch_for_res(batch_size=128):
    inputs_images = []   
    codes = []
    max_width_image = 0
    info = ""
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        font_length = random.randint(3, 80)
        if random.random()>0.5:
            font_size = random.randint(8, 49)    
        else:
            font_size = random.randint(8, 15) 
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,2,3,4,5])     #删除了2
        while True:
            text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)
            image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint )
            image = utils_pil.resize_by_height(image, image_height, random.random()>0.5)
            w, h = image.size
            if w * h < image_size * image_size: break

        image = utils_font.add_noise(image)   
        image = utils_pil.convert_to_gray(image)                   
        image = np.asarray(image)     
        # image = utils.resize(image, height=image_height)
        image = image * random.uniform(0.3, 1)        
        if random.random()>0.5:
            image = (255. - image) / 255.
        else:
            image = image / 255.
        inputs_images.append(image)
        codes.append([CHARS.index(char) for char in text])                  

        info = info+"%s\n\r" % utils_font.get_font_url(text, font_name, font_size, font_mode, font_hint)

    inputs = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        inputs[i,:] = utils.img2img(inputs_images[i],np.zeros([image_size, image_size]))

    labels = [np.asarray(i) for i in codes]
    sparse_labels = utils.sparse_tuple_from(labels)
    seq_len = np.ones(batch_size) * (image_size * image_size ) // (POOL_SIZE * POOL_SIZE)                
    return inputs, sparse_labels, seq_len, info


def train():
    inputs, labels, global_step, \
        res_loss, res_optim, seq_len, res_acc, res_decoded, \
        net_g = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_G_dir = os.path.join(model_dir, "FG")
    model_R_dir = os.path.join(model_dir, "FR")

    if not os.path.exists(model_R_dir): os.mkdir(model_R_dir)
    if not os.path.exists(model_G_dir): os.mkdir(model_G_dir)  
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        r_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RES'), sharded=True)
        g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g'), sharded=True)

        ckpt = tf.train.get_checkpoint_state(model_G_dir)
        if ckpt and ckpt.model_checkpoint_path:           
            print("Restore Model G...")
            g_saver.restore(session, ckpt.model_checkpoint_path)   

        ckpt = tf.train.get_checkpoint_state(model_R_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model R...")
            r_saver.restore(session, ckpt.model_checkpoint_path)    

        while True:
            errA = errD1 = errD2 = 1
            for batch in range(BATCHES):
                train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(16)

                start = time.time() 
                p_net_g = session.run(net_g, {inputs: train_inputs}) 
                p_net_g = np.squeeze(p_net_g)
                feed = {inputs: p_net_g, labels: train_labels, seq_len: train_seq_len}  
                errR, acc, _ , steps= session.run([res_loss, res_acc, res_optim, global_step], feed)
                print("%d time: %4.4fs, res_loss: %.8f, res_acc: %.8f " % (steps, time.time() - start, errR, acc))
                if np.isnan(errR) or np.isinf(errR) :
                    print("Error: cost is nan or inf")
                    return                      

                # 报告
                if steps > 0 and steps % REPORT_STEPS == 0:
                    train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(4)   
                    print(train_info)          
                    p_net_g = session.run(net_g, {inputs: train_inputs}) 
                    p_net_g = np.squeeze(p_net_g)
                    decoded_list = session.run(res_decoded[0], {inputs: p_net_g, seq_len: train_seq_len}) 

                    for i in range(4): 
                        _img = np.vstack((train_inputs[i], p_net_g[i])) 
                        cv2.imwrite(os.path.join(curr_dir,"test","%s_%s.png"%(steps,i)), _img * 255) 

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

            print("Save Model R ...")
            r_saver.save(session, os.path.join(model_R_dir, "R.ckpt"), global_step=steps)
            try:
                ckpt = tf.train.get_checkpoint_state(model_G_dir)
                if ckpt and ckpt.model_checkpoint_path:           
                    print("Restore Model G...")
                    g_saver.restore(session, ckpt.model_checkpoint_path)   
            except:
                pass

if __name__ == '__main__':
    train()