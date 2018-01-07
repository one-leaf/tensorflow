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

curr_dir = os.path.dirname(__file__)

image_height = 16
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

BATCHES = 64
BATCH_SIZE = 4
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_ascii_srgan"

# 位置调整
def neural_networks_trim():
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size], name="inputs")
    targets = tf.placeholder(tf.float32, [None, image_size, image_size], name="targets")

    global_step = tf.Variable(0, trainable=False)

    layer = tf.reshape(inputs, (-1, image_size, image_size, 1))
    for cnn in (64,128,256,512,512,512,512,1024):
        layer = slim.conv2d(layer, cnn, [3,3], stride=2)        
    print(layer.shape)
    layer = slim.flatten(layer)
    print(layer.shape)
    layer = slim.fully_connected(layer,1024, activation_fn=None)
    layer = tf.reshape(layer, (-1, 1, 1, 1024))   
    for cnn in (512,512,512,512,256,128,64,1):  
        layer = slim.conv2d_transpose(layer, cnn, [3,3], stride=2)
    print(layer.shape)

    logits = tf.reshape(layer, (-1, image_size, image_size))   
    loss = tf.losses.mean_squared_error(logits, targets)   
    optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(loss, global_step=global_step)
    
    return  inputs, targets, global_step, logits, loss, optim

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

# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch_for_gan(batch_size=128):
    input_images  = []
    trim_images = []
    clear_images = []
    half_clear_images = []
    max_width_image = 0
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        font_size = image_height #random.randint(image_height, 64)    
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,2,3,4,5])     #删除了2
        while True:
            font_length = random.randint(3, 400)
            text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)
            image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint)
            image = utils_pil.resize_by_height(image, image_height)
            w, h = image.size
            if w * h <= image_size * image_size: break
        image = utils_pil.convert_to_gray(image)

        # 干净的图片，给降噪网络用
        clears_image = image.copy()
        clears_image = np.asarray(clears_image)
        clears_image = (255. - clears_image) / 255. 
        clear_images.append(clears_image)

        _h =  random.randint(9, image_height+1)
        image = utils_pil.resize_by_height(image, _h)        
        image = utils_pil.resize_by_height(image, image_height, random.random()>0.5) 
        
        # 给早期降噪网络使用
        half_clear_image = image.copy()
        half_clear_image = utils_font.add_noise(half_clear_image) 
        half_clear_image = np.asarray(half_clear_image)
        half_clear_image = half_clear_image * random.uniform(0.3, 1)
        if random.random()>0.5:
            half_clear_image = (255. - half_clear_image) / 255.
        else:
            half_clear_image = half_clear_image / 255.           
        half_clear_images.append(half_clear_image) 

        # 随机移动位置并缩小 trims_image 为字体实际位置标识
        image, trims_image = utils_pil.random_space2(image)
        trims_image = np.asarray(trims_image)
        trims_image = (255. - trims_image) / 255.         
        trim_images.append(trims_image)

        image = utils_font.add_noise(image)   
        image = np.asarray(image)
        image = image * random.uniform(0.3, 1)
        if random.random()>0.5:
            image = (255. - image) / 255.
        else:
            image = image / 255.           
        input_images.append(image)   

    inputs = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        inputs[i,:] = utils.img2img(input_images[i], np.zeros([image_size, image_size]), image_height)

    trims = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        trims[i,:] = utils.img2img(trim_images[i], np.zeros([image_size, image_size]), image_height)

    clears = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        clears[i,:] = utils.img2img(clear_images[i], np.zeros([image_size, image_size]), image_height)

    half_clears = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        half_clears[i,:] = utils.img2img(half_clear_images[i], np.zeros([image_size, image_size]), image_height)

    return inputs, trims, clears, half_clears

def train():
    inputs, targets, global_step, logits, loss, optim = neural_networks_trim()
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        while True:
            for batch in range(BATCHES):
                batch_size = 16
                train_inputs, train_trims, train_clears, train_half_clears = get_next_batch_for_gan(batch_size)
                feed = {inputs: train_inputs, targets: train_trims}
                start = time.time()                
                err, _, steps, net = session.run([loss, optim, global_step, logits], feed)
                print("T %d time: %4.4fs, loss: %.8f" % (steps, time.time() - start, err))

                # 报告
                if steps > 0 and steps % REPORT_STEPS ==0:
                    for i in range(batch_size): 
                        _img = np.vstack((train_inputs[i], net[i], train_trims[i])) 
                        cv2.imwrite(os.path.join(curr_dir,"test","T%s_%s.png"%(steps,i)), _img * 255) 
            
if __name__ == '__main__':
    train()