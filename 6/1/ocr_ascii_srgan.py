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
import utils_pil, utils_font

curr_dir = os.path.dirname(__file__)

image_height = 32

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

BATCHES = 10
BATCH_SIZE = 4
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_ascii_srgan"

# 模型来源，有改动 http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
# 取消了pool1,增加了3次pool，删除了最后的avg和fc层
def RESNET50(inputs, isKeepSize=False):
    # 1
    layer = slim.conv2d(inputs, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    # 2a
    templayer = layer
    layer = slim.conv2d(layer, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    templayer = slim.conv2d(templayer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 2b
    templayer = layer
    layer = slim.conv2d(layer, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 2c
    templayer = layer
    layer = slim.conv2d(layer, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 3a
    if isKeepSize:
        layer = slim.max_pool2d(layer,[2,2], stride=1, padding="SAME")
    else:    
        layer = slim.max_pool2d(layer,[2,2])
    templayer = layer
    layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    templayer = slim.conv2d(templayer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 3b
    templayer = layer
    layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 3c
    templayer = layer
    layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 3d
    templayer = layer
    layer = slim.conv2d(layer, 128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 128, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 4a
    if isKeepSize:
        layer = slim.max_pool2d(layer,[2,2], stride=1, padding="SAME")
    else:    
        layer = slim.max_pool2d(layer,[2,2])
    templayer = layer
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    templayer = slim.conv2d(templayer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 4b
    templayer = layer
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 4c
    templayer = layer
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 4d
    templayer = layer
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)        
    # 4e
    templayer = layer
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer) 
    # 4f  
    templayer = layer
    layer = slim.conv2d(layer, 256, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 1024, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer) 
    # 5a        
    if isKeepSize:
        layer = slim.max_pool2d(layer,[2,2], stride=1, padding="SAME")
    else:    
        layer = slim.max_pool2d(layer,[2,2])
    conv = layer
    templayer = layer
    layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 512, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 2048, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    templayer = slim.conv2d(templayer, 2048, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 5b
    templayer = layer
    layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 512, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 2048, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    # 5c
    templayer = layer
    layer = slim.conv2d(layer, 512, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 512, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 2048, [1,1], normalizer_fn=slim.batch_norm, activation_fn=None)
    layer = tf.nn.relu(templayer+layer)
    return layer, conv    

## 增加残差网络
# def addResLayer(inputs):
#     layer = slim.conv2d(inputs, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
#     layer = slim.conv2d(inputs, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
#     outputs = inputs + layer
#     return outputs  

# 参考 https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py
def INCEPTIONV3(inputs):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        # mixed_5b
        net    = inputs
        layer0 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,    48, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 64, [5,5], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net, [3,3])
        layer3 = slim.conv2d(layer3, 32, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_5c => 288
        layer0 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,    48, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 64, [5,5], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)        
        # mixed_5d => 288
        layer0 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,    48, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 64, [5,5], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3, 64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net = tf.concat([layer0, layer1, layer2, layer3], 3)   
        # mixed_6a => 768
        layer0 = slim.conv2d(net,   384, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,    64, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1, 96, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.max_pool2d(net,  [3,3])
        net    = tf.concat([layer0, layer1, layer2], 3)
        # mixed_6b => 768
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,128, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,   128, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,128, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,128, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,128, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_6c => 768
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,160, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,   160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_6d => 768
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,160, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,   160, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,160, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_6e => 768
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_7a => 1280
        layer0 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer0 = slim.conv2d(layer0,320, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [1,7], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [7,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(layer1,192, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.max_pool2d(net,  [3,3])
        net    = tf.concat([layer0, layer1, layer2], 3)
        # mixed_7b => 2048
        layer0 = slim.conv2d(net,   320, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   384, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1_1 = slim.conv2d(layer1,384, [1,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1_2 = slim.conv2d(layer1,384, [3,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1    = tf.concat([layer1_1, layer1_2], 3)
        layer2 = slim.conv2d(net,   448, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,   384, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2_1 = slim.conv2d(layer2,384, [1,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2_2 = slim.conv2d(layer2,384, [3,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2    = tf.concat([layer2_1, layer2_2], 3)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
        # mixed_7c => 2048
        layer0 = slim.conv2d(net,   320, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1 = slim.conv2d(net,   384, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1_1 = slim.conv2d(layer1,384, [1,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1_2 = slim.conv2d(layer1,384, [3,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer1    = tf.concat([layer1_1, layer1_2], 3)
        layer2 = slim.conv2d(net,   448, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2 = slim.conv2d(layer2,   384, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2_1 = slim.conv2d(layer2,384, [1,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2_2 = slim.conv2d(layer2,384, [3,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        layer2    = tf.concat([layer2_1, layer2_2], 3)
        layer3 = slim.avg_pool2d(net,  [3,3])
        layer3 = slim.conv2d(layer3,192, [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        net    = tf.concat([layer0, layer1, layer2, layer3], 3)
    return net

def old_SRGAN_g(inputs):
    layer = slim.conv2d(inputs, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    temp = layer
    # B residual blocks
    for i in range(16):
        layer = addResLayer(layer)
    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn = slim.batch_norm, activation_fn = None)
    layer = layer + temp        
    # B residual blacks end
    layer = slim.conv2d(layer, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 256, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    layer = slim.conv2d(layer, 1,   [1,1], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.tanh)
    return layer

# 原参考 https://github.com/zsdonghao/SRGAN/blob/master/model.py 失败，后期和D对抗时无法提升，后改为 resnet50
def SRGAN_g(inputs, reuse=False):    
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:      
        # layer, _ = RESNET50(inputs, True)
        layer = INCEPTIONV3(inputs)
        layer = slim.conv2d(layer, 1,   [1,1], normalizer_fn=None, activation_fn=None)
        return layer

def SRGAN_d(inputs, reuse=False):
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        layer, _ = RESNET50(inputs)        
        layer = slim.fully_connected(layer, 1024, activation_fn=tf.nn.relu)
        logits = slim.fully_connected(layer, 1, activation_fn=tf.identity)
        return logits

def RES(inputs, reuse = False):
    with tf.variable_scope("RES", reuse=reuse):
        layer, conv = RESNET50(inputs)
        layer = slim.conv2d(layer, CLASSES_NUMBER, [1,1], normalizer_fn=None, activation_fn=None)
        shape = tf.shape(inputs)
        batch_size, image_width = shape[0], shape[1]        
        layer = tf.reshape(layer, [batch_size, -1, CLASSES_NUMBER])
        return layer, conv

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, None, image_height], name="inputs")
    targets = tf.placeholder(tf.float32, [None, None, image_height], name="targets")
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    global_step = tf.Variable(0, trainable=False)

    shape = tf.shape(inputs)
    batch_size, image_width = shape[0], shape[1]

    layer = tf.reshape(inputs, (batch_size, image_width, image_height, 1))
    layer_targets = tf.reshape(targets, (batch_size, image_width, image_height, 1))

    net_g = SRGAN_g(layer, reuse = False)
    logits_real = SRGAN_d(layer_targets, reuse = False)
    logits_fake = SRGAN_d(net_g, reuse = True)

    # net_res, _ = RES(layer_targets, reuse = False)
    net_res, _ = RES(net_g, reuse = False)
    seq_len = tf.placeholder(tf.int32, [None])
    res_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RES')
    # 需要变换到 time_major == True [max_time x batch_size x num_classes]
    net_res = tf.transpose(net_res, (1, 0, 2))
    res_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=net_res, sequence_length=seq_len))
    res_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(res_loss, global_step=global_step, var_list=res_vars)
    res_decoded, _ = tf.nn.ctc_beam_search_decoder(net_res, seq_len, beam_width=10, merge_repeated=False)
    res_acc = tf.reduce_sum(tf.edit_distance(tf.cast(res_decoded[0], tf.int32), labels, normalize=False))
    res_acc = 1 - res_acc / tf.to_float(tf.size(labels.values))


    _, res_target_emb   = RES(layer_targets, reuse = True)
    _, res_predict_emb  = RES(net_g, reuse = True)

    d_loss1 = 1e-3 * tf.losses.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
    d_loss2 = -1e-3 * tf.losses.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
    d_loss  = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tf.losses.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
    g_mse_loss = tf.losses.mean_squared_error(net_g, layer_targets)
    g_res_loss = tf.losses.mean_squared_error(res_target_emb, res_predict_emb)
    g_loss     = g_gan_loss + g_mse_loss + g_res_loss
    
    g_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g')
    d_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_d')

    g_optim_mse = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_mse_loss, global_step=global_step, var_list=g_vars)

    g_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_loss, global_step=global_step, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(d_loss, global_step=global_step, var_list=d_vars)

    return inputs, targets, labels, global_step, g_optim_mse, d_loss, d_loss1, d_loss2, d_optim, \
            g_loss, g_mse_loss, g_res_loss, g_gan_loss, g_optim, net_g, res_loss, res_optim, seq_len, res_acc, res_decoded


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
    images = []   
    codes = []
    max_width_image = 0
    info = ""
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        font_length = random.randint(25, 30)
        if random.random()>0.5:
            font_size = random.randint(8, 49)    
        else:
            font_size = random.randint(8, 15) 
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,3,4,5])     #删除了2
        text = random.sample(CHARS, 12)
        text = text+text+[" "," "]
        random.shuffle(text)
        text = "".join(text).strip()
        codes.append([CHARS.index(char) for char in text])          
        image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint )
        image = utils_pil.resize_by_height(image, image_height, random.random()>0.5)
        image = utils_font.add_noise(image)   
        image = utils_pil.convert_to_gray(image)                   
        image = np.asarray(image)     
        image = utils.resize(image, height=image_height)
        image = (255. - image) / 255.
        images.append(image)
        if image.shape[1] > max_width_image: 
            max_width_image = image.shape[1]
        info = info+"%s\n\r" % utils_font.get_font_url(text, font_name, font_size, font_mode, font_hint)
    max_width_image = max_width_image + (POOL_SIZE - max_width_image % POOL_SIZE)
    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = utils.img2vec(images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.transpose(image_vec)

    labels = [np.asarray(i) for i in codes]
    sparse_labels = utils.sparse_tuple_from(labels)
    seq_len = np.ones(batch_size) * (max_width_image * image_height ) // (POOL_SIZE * POOL_SIZE)                
    return inputs, sparse_labels, seq_len, info
     

# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch_for_srgan(batch_size=128):
    images = []   
    to_images = []
    max_width_image = 0
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        font_length = random.randint(25, 30)
        font_size = 36 #random.randint(image_height, 64)    
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,3,4,5])     #删除了2
        text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)
        image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint)
        image = utils_pil.resize_by_height(image, image_height)
        to_image = image.copy()

        _h =  random.randint(9, image_height // random.choice([1,1.5,2,2.5]))
        image = utils_font.add_noise(image)   
        image = utils_pil.convert_to_gray(image)            
        image = utils_pil.resize_by_height(image, _h)        
        image = utils_pil.resize_by_height(image, image_height, random.random()>0.5)        
        image = np.asarray(image)
        image = utils.resize(image, height=image_height)
        image = (255. - image) / 255.
        images.append(image)

        to_image = utils_pil.convert_to_gray(to_image)
        to_image = np.asarray(to_image)   
        to_image = utils.resize(to_image, height=image_height)
        to_image = utils.img2bwinv(to_image)
        to_image = to_image / 255.        
        to_images.append(to_image)

        if image.shape[1] > max_width_image: 
            max_width_image = image.shape[1]
        if to_image.shape[1] > max_width_image: 
            max_width_image = to_image.shape[1]      

    max_width_image = max_width_image + (POOL_SIZE - max_width_image % POOL_SIZE)
    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = utils.img2vec(images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.transpose(image_vec)

    targets = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(to_images)):
        image_vec = utils.img2vec(to_images[i], height=image_height, width=max_width_image, flatten=False)
        targets[i,:] = np.transpose(image_vec)

    return inputs, targets

def train():
    inputs, targets, labels, global_step, g_optim_mse, d_loss, d_loss1, d_loss2, d_optim, \
        g_loss, g_mse_loss, g_res_loss, g_gan_loss, g_optim, net_g, \
        res_loss, res_optim, seq_len, res_acc, res_decoded = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_R_dir = os.path.join(model_dir, "R")
    model_D_dir = os.path.join(model_dir, "D")
    model_G_dir = os.path.join(model_dir, "G")
    if not os.path.exists(model_R_dir): os.mkdir(model_R_dir)
    if not os.path.exists(model_D_dir): os.mkdir(model_D_dir)
    if not os.path.exists(model_G_dir): os.mkdir(model_G_dir)  
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        
        r_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RES'), max_to_keep=5)
        d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_d'), max_to_keep=5)
        g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g'), max_to_keep=5)

        ckpt = tf.train.get_checkpoint_state(model_G_dir)
        if ckpt and ckpt.model_checkpoint_path:           
            print("Restore Model G...")
            g_saver.restore(session, ckpt.model_checkpoint_path)   
        ckpt = tf.train.get_checkpoint_state(model_R_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model R...")
            r_saver.restore(session, ckpt.model_checkpoint_path)   
        ckpt = tf.train.get_checkpoint_state(model_D_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model D...")
            d_saver.restore(session, ckpt.model_checkpoint_path)    
 
        while True:
            for batch in range(BATCHES):
                train_inputs, train_targets = get_next_batch_for_srgan(1)
                feed = {inputs: train_inputs, targets: train_targets}

                # train GAN (SRGAN)
                start = time.time()                
                ## update D
                errD, errD1, errD2, _, steps = session.run([d_loss, d_loss1, d_loss2, d_optim, global_step], feed)
                print("%d time: %4.4fs, d_loss: %.8f (d_loss1: %.6f  d_loss2: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))
                if np.isnan(errD) or np.isinf(errD):
                    print("Error: cost is nan or inf")
                    return 
                start_steps = steps     

                ## update G
                start = time.time()                                
                errG, errM, errV, errA, _, steps = session.run([g_loss, g_mse_loss, g_res_loss, g_gan_loss, g_optim, global_step], feed)
                print("%d time: %4.4fs, g_loss: %.8f (mse: %.6f res: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errV, errA))
                if np.isnan(errG) or np.isinf(errG) or np.isnan(errM) or np.isinf(errM) or np.isnan(errA) or np.isinf(errA):
                    print("Error: cost is nan or inf")
                    return 

                # 如果平均差太高，单独学习降低平均差
                if (errM > 0.5 and errD > 10) or errM > 10:   
                    for i in range(16):
                        train_inputs, train_targets = get_next_batch_for_srgan(1)
                        feed = {inputs: train_inputs, targets: train_targets}
                        # train G
                        start = time.time() 
                        errM, _ , steps= session.run([g_mse_loss, g_optim_mse, global_step], feed)
                        print("%d time: %4.4fs, g_mse_loss: %.8f " % (steps, time.time() - start, errM))
                        if np.isnan(errM) or np.isinf(errM) :
                            print("Error: cost is nan or inf")
                            return

                # 如果D网络的差异太大，需要多学习下G网络
                if errD < -10:
                   for i in range(16):
                        train_inputs, train_targets = get_next_batch_for_srgan(1)
                        feed = {inputs: train_inputs, targets: train_targets}

                        ## update G
                        start = time.time()                                
                        errG, errM, errV, errA, _, steps = session.run([g_loss, g_mse_loss, g_res_loss, g_gan_loss, g_optim, global_step], feed)
                        print("%d time: %4.4fs, g_loss: %.8f (mse: %.6f res: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errV, errA))
                        if np.isnan(errG) or np.isinf(errG) or np.isnan(errA) or np.isinf(errA):
                            print("Error: cost is nan or inf")
                            return 

                # 如果图片相差不远，训练RES
                if errM < 0.5:
                    # train res
                    for i in range(16):
                        train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(4)
                        feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len}
                        start = time.time() 
                        errR, acc, _ , steps= session.run([res_loss, res_acc, res_optim, global_step], feed)
                        print("%d time: %4.4fs, res_loss: %.8f, res_acc: %.8f " % (steps, time.time() - start, errR, acc))
                        if np.isnan(errR) or np.isinf(errR) :
                            print("Error: cost is nan or inf")
                            return                       
                        if errR > 15 and acc > 0.8: print(train_info)

                if steps > 0 and steps % REPORT_STEPS < (steps-start_steps):
                    train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res(4)   
                    print(train_info)          
                    feed = {inputs: train_inputs, seq_len: train_seq_len}
                    b_predictions, decoded_list = session.run([net_g, res_decoded[0]], feed) 
                    for i in range(4):                    
                        _predictions = np.reshape(b_predictions[i],train_inputs[i].shape)   
                        _pred = np.transpose(_predictions)   
                        _img = np.vstack((np.transpose(train_inputs[i]), _pred)) 
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
            print("Save Model D ...")
            d_saver.save(session, os.path.join(model_D_dir, "D.ckpt"), global_step=steps)
            print("Save Model G ...")
            g_saver.save(session, os.path.join(model_G_dir, "G.ckpt"), global_step=steps)
                
if __name__ == '__main__':
    train()