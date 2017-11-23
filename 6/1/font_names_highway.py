# coding=utf-8
# 中文OCR学习，尝试多层

import tensorflow as tf
import numpy as np
import os
from utils import readImgFile, img2gray, img2bwinv, img2vec, dropZeroEdges, resize, save, getImage
import time
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import tensorflow.contrib.slim as slim
import math

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

#初始化学习速率
LEARNING_RATE_INITIAL = 1e-3
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
    labels = tf.placeholder(tf.int32,[None], name="labels")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    drop_prob = 1 - keep_prob

    shape = tf.shape(inputs)
    batch_size, image_width = shape[0], shape[1]

    layer = tf.reshape(inputs, [batch_size,image_width,image_height,1])

    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm)
    for i in range(POOL_COUNT):
        for j in range(10):
            layer = addHighwayLayer(layer)
        layer = slim.conv2d(layer, 64, [3,3], stride=[2, 2], normalizer_fn=slim.batch_norm)  
    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=None)

    # prediction = slim.layers.softmax(slim.layers.flatten(layer))

    layer = tf.reshape(layer,[batch_size, 64 * image_width * image_height // POOL_SIZE // POOL_SIZE])
    layer = slim.fully_connected(layer, 512, normalizer_fn=slim.batch_norm)
    layer = slim.fully_connected(layer, num_classes, normalizer_fn=slim.batch_norm, activation_fn=None)

    predictions = slim.layers.softmax(layer)

    target = tf.one_hot(tf.cast(labels, tf.int32), num_classes, 1, 0)

    loss = tf.losses.softmax_cross_entropy(target, predictions) 
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_INITIAL)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    return inputs, labels, predictions, keep_prob, loss, train_op

FontDir = os.path.join(curr_dir,"fonts")
FontNames = sorted([os.path.join(FontDir, name) for name in os.listdir(FontDir)])
num_classes = len(FontNames)

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 
# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch(batch_size=128):
    images = []   
    font_names = []
    max_width_image = 0
    font_min_length = random.randint(50, 60)
    for i in range(batch_size):
        font_name = random.choice(FontNames)
        font_length = random.randint(font_min_length-5, font_min_length+5)
        font_size = random.randint(9, 64)        
        text, image= getImage(CHARS, font_name, image_height, font_length, font_size, eng_world_list)
        images.append(image)
        if image.shape[1] > max_width_image: 
            max_width_image = image.shape[1]
        font_names.append(FontNames.index(font_name))

    # 凑成4的整数倍
    max_width_image = max_width_image + (POOL_SIZE - max_width_image % POOL_SIZE)
    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = img2vec(images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.transpose(image_vec)

    return inputs, font_names

def train():
    inputs, labels, predictions, keep_prob, loss, train_op = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, "model_fontnames_highway")
    if not os.path.exists(model_dir): os.mkdir(model_dir)

    slim.learning.train(train_op, my_log_dir)                                   

    

    

if __name__ == '__main__':
    train()