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

def RES(inputs, reuse = False):
    with tf.variable_scope("RES", reuse=reuse):
        layer = utils_nn.resNet50(inputs, True)
        # shape = tf.shape(inputs)
        # batch_size = shape[0] 
        layer = slim.flatten(layer) 
        layer = slim.fully_connected(layer, 1000, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer = slim.fully_connected(layer, 4, normalizer_fn=None, activation_fn=None)  
        return layer

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,256,256]
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size], name="inputs")
    labels = tf.placeholder(tf.float32, [None, 4], name="labels")
    global_step = tf.Variable(0, trainable=False)

    layer = tf.reshape(inputs, (-1, image_size, image_size, 1))

    net_res = RES(layer, reuse = False)

    res_loss = tf.reduce_sum(tf.square(labels - net_res))
    res_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(res_loss, global_step=global_step)
    
    return  inputs, labels, global_step, net_res, res_loss, res_optim

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

def get_next_batch_for_res(batch_size=128, add_noise=True):
    inputs_images = []   
    codes = []
    max_width_image = 0
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        if random.random()>0.5:
            font_size = random.randint(8, 49)    
        else:
            font_size = random.randint(8, 15) 
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,2,3,4,5])     #删除了2
        while True:
            font_length = random.randint(5, 40)

            text = random.sample(CHARS, font_length)
            text = text+text+[" "," "]
            random.shuffle(text)
            text = "".join(text).strip()

            #text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)
            image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint )
            if add_noise:                  
                image = utils_pil.resize_by_height(image, image_height, random.random()>0.5)
            else:
                image = utils_pil.resize_by_height(image, image_height)
            w, h = image.size
            if w * h < image_size * image_size: break

        image = utils_pil.convert_to_gray(image) 

        image,x,y,w,h = utils_pil.random_space(image)

        if add_noise:                  
            image = utils_font.add_noise(image)   
        image = np.asarray(image)     
        # image = utils.resize(image, height=image_height)
        if add_noise:
            image = image * random.uniform(0.3, 1)        

        if add_noise and random.random()>0.5:
            image = image / 255.
        else:
            image = (255. - image) / 255.

        inputs_images.append(image)
        codes.append([x,y,w,h])                  

    inputs = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        inputs[i,:] = utils.img2img(inputs_images[i],np.zeros([image_size, image_size]))

    labels = [np.asarray(i) for i in codes]
    return inputs, labels


def train():
    inputs, labels, global_step, net_res, res_loss, res_optim = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_R_dir = os.path.join(model_dir, "T")

    if not os.path.exists(model_R_dir): os.mkdir(model_R_dir)
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        r_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RES'), sharded=True)

        ckpt = tf.train.get_checkpoint_state(model_R_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model R...")
            r_saver.restore(session, ckpt.model_checkpoint_path)    

        while True:
            errA = errD1 = errD2 = 1
            for batch in range(BATCHES):
                train_inputs, train_labels = get_next_batch_for_res(4, False)

                start = time.time() 

                feed = {inputs: train_inputs, labels: train_labels} 
                errR, _ , steps= session.run([res_loss, res_optim, global_step], feed)
                print("%d time: %4.4fs, res_loss: %.8f" % (steps, time.time() - start, errR))
                if np.isnan(errR) or np.isinf(errR) :
                    print("Error: cost is nan or inf")
                    return                      

                # 报告
                if steps > 0 and steps % REPORT_STEPS == 0:
                    train_inputs, train_labels = get_next_batch_for_res(4)   
                    decoded_list = session.run(net_res, {inputs: train_inputs}) 

                    for i in range(4): 
                        cv2.imwrite(os.path.join(curr_dir,"test","T%s_%s.png"%(steps,i)), train_inputs[i] * 255) 

                    for idx in range(min(len(train_labels),len(decoded_list))):
                        number = train_labels[idx]
                        detect_number = decoded_list[idx]  
                        hit = (number == detect_number)          
                        print("%6s" % hit, number)
                        print("%6s" % "",  detect_number)


            print("Save Model R ...")
            r_saver.save(session, os.path.join(model_R_dir, "T.ckpt"), global_step=steps)


if __name__ == '__main__':
    train()