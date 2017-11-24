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
LEARNING_RATE_INITIAL = 1e-4
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 500
MOMENTUM = 0.9

BATCHES = 64
BATCH_SIZE = 10
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE


# 增加 Highway 网络
def addHighwayLayer(inputs):
    H = slim.conv2d(inputs, 64, [3,3])
    T = slim.conv2d(inputs, 64, [3,3], biases_initializer = tf.constant_initializer(-1.0), activation_fn=tf.nn.sigmoid)    
    outputs = H * T + inputs * (1.0 - T)
    return outputs    

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, None, image_height], name="inputs")
    labels = tf.placeholder(tf.float32, [None, None, image_height], name="labels")
    
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    drop_prob = 1 - keep_prob

    shape = tf.shape(inputs)
    batch_size, image_width = shape[0], shape[1]

    layer = tf.reshape(inputs, [batch_size,image_width,image_height,1])

    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm)
    for i in range(5):
        for j in range(5):
            layer = addHighwayLayer(layer)
        layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm)  
    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=None)

    layer = tf.layers.dense(layer, 64, activation=tf.nn.relu) #(batch_size, image_width, image_height, 64)
    layer = tf.reshape(layer,(-1,image_height*64))
    predictions = tf.layers.dense(layer, image_height, activation=tf.nn.relu)
    predictions = tf.reshape(predictions,(batch_size,image_width,image_height))

    _labels = tf.reshape(labels,(batch_size,-1))
    _predictions = tf.reshape(predictions,(batch_size,-1))
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(_predictions - _labels)))

    return inputs, labels, predictions, keep_prob, loss

FontDir = os.path.join(curr_dir,"fonts")
FontNames = []    
for name in os.listdir(FontDir):
    fontName = os.path.join(FontDir, name)
    if fontName.lower().endswith('ttf') or \
        fontName.lower().endswith('ttc') or \
        fontName.lower().endswith('otf'):
        FontNames.append(fontName)
ConsolasFont = os.path.join(curr_dir,"fonts","Consolas.ttf")

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 
# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch(batch_size=128):
    images = []   
    to_images = []
    max_width_image = 0
    font_min_length = random.randint(10, 20)
    for i in range(batch_size):
        font_name = random.choice(FontNames)
        font_length = random.randint(font_min_length-5, font_min_length+5)
        font_size = random.randint(9, 64)        
        text, image= utils.getImage(CHARS, font_name, image_height, font_length, font_size, eng_world_list)
        images.append(image)
        if images.shape[0] > max_width_image: 
            max_width_image = images.shape[0]
        to_image=utils.renderNormalFontByPIL(ConsolasFont,64,text)
        to_image=utils.trim(to_image)
        to_image=to_image.resize((max_width_image, image_height), Image.ANTIALIAS)
        to_image=np.asarray(to_image)
        #to_image=utils.resize(to_image, height=image_height)
        to_image=utils.img2gray(to_image)
        to_image=to_image / 255
        to_images.append(to_image)

    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = utils.img2vec(images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.transpose(image_vec)

    labels = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(to_images)):
        image_vec = utils.img2vec(to_images[i], height=image_height, width=max_width_image, flatten=False)
        labels[i,:] = np.transpose(image_vec)
    return inputs, labels

def train():
    global_step = tf.Variable(0, trainable=False)
    inputs, labels, predictions, keep_prob, loss = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, "model_font2font_highway")
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    saver_prefix = os.path.join(model_dir, "model.ckpt")        

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_INITIAL)
    train_op = optimizer.minimize(loss, global_step=global_step)
   
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.Saver(max_to_keep=5)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model ...")
            saver.restore(session, ckpt.model_checkpoint_path)    
        while True:
            for batch in range(BATCHES):
                start = time.time()                
                train_inputs, train_labels = get_next_batch(BATCH_SIZE)             
                feed = {inputs: train_inputs, labels: train_labels, keep_prob: 0.95}
                b_loss, b_labels, b_predictions,  steps,  _ = \
                    session.run([loss, labels, predictions, global_step, train_op], feed)

                seconds = round(time.time() - start,2)
                print("step:", steps, "cost:", b_loss, "batch seconds:", seconds)
                if np.isnan(b_loss) or np.isinf(b_loss):
                    print("Error: cost is nan or inf")
                    train_labels_list = decode_sparse_tensor(train_labels)
                    for i, train_label in enumerate(train_labels_list):
                        print(i,list_to_chars(train_label))
                    return   
                
                if seconds > 60: 
                    print('Exit for long time')
                    return


            saver.save(session, saver_prefix, global_step=steps)
                
if __name__ == '__main__':
    train()