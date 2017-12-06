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
import utils_pil, utils_font

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
BATCH_SIZE = 16
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_font2font_res"

# 增加残差网络
def addResLayer(inputs, rate=3):
    layer = slim.batch_norm(inputs, activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.conv2d(layer, 64, [rate,rate],activation_fn=None)
    layer = slim.batch_norm(layer, activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.conv2d(layer, 64, [rate,rate],activation_fn=None)
    outputs = inputs + layer
    return outputs    

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, None, image_height], name="inputs")
    labels = tf.placeholder(tf.float32, [None, None, image_height], name="labels")
    
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    drop_prob = 1 - keep_prob

    shape = tf.shape(inputs)
    batch_size, image_width = shape[0], shape[1]

    layer = tf.reshape(inputs, (batch_size, image_width, image_height, 1))

    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm)
    for i in range(POOL_COUNT):
        for j in range(3):
            layer = addResLayer(layer)
        layer = slim.conv2d(layer, 64, [3,3], stride=[1, 1], normalizer_fn=slim.batch_norm)  
        
    predictions = slim.conv2d(layer, 1, [1,1], normalizer_fn = None, activation_fn = tf.nn.tanh)
    predictions = tf.reshape(predictions, (batch_size, image_width, image_height, 1 ))

    # _labels = tf.reshape(labels, (batch_size, image_width, image_height, 1 ))
    # _predictions = slim.repeat(predictions, 2, slim.max_pool2d, [2, 2])
    # _labels = slim.repeat(_labels, 2, slim.max_pool2d, [2, 2])

    _predictions = tf.layers.flatten(predictions)
    _labels = tf.layers.flatten(labels)
    loss = tf.square(_predictions - _labels)
    fill = tf.ones_like(loss) * 0
    loss_filled = tf.where(tf.is_finite(loss), loss, fill)
    finite_count = tf.reduce_sum(tf.cast(tf.is_finite(loss), tf.float32))
    mse = tf.reduce_sum(loss_filled) / finite_count
    loss = mse/2.
    return inputs, labels, predictions, keep_prob, loss


ENGFontNames, CHIFontNames = utils_font.get_font_names_from_url()
print("EngFontNames", ENGFontNames)
print("CHIFontNames", CHIFontNames)
AllFontNames = ENGFontNames + CHIFontNames

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 
# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch(batch_size=128):
    images = []   
    to_images = []
    max_width_image = 0
    font_min_length = random.randint(10, 20)
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        # font_length = random.randint(font_min_length-5, font_min_length+5)
        font_length = random.randint(3, 5)
        font_size = random.randint(image_height, 64)    
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,2,3,4,5])     
        text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)          
        image = utils_font.get_font_image_from_url(text, font_name ,font_size, fontmode = font_mode, fonthint = font_hint )
        to_image = image.copy()
        image = utils_font.add_noise(image)   
        image = utils_pil.convert_to_gray(image)
        rate =  random.randint(8, 17) / font_size
        image = utils_pil.resize(image, rate)
        image = np.asarray(image)     
        image = utils.resize(image, height=image_height)
        image = (255. - image) / 255.
        images.append(image)

        # to_image = utils_font.get_font_image_from_url(text, font_name ,image_height, fontmode = font_mode, fonthint = font_hint)
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

    labels = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(to_images)):
        image_vec = utils.img2vec(to_images[i], height=image_height, width=max_width_image, flatten=False)
        labels[i,:] = np.transpose(image_vec)
    return inputs, labels

def train():
    global_step = tf.Variable(0, trainable=False)
    inputs, labels, predictions, keep_prob, loss = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
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

                if steps > 0 and steps % REPORT_STEPS == 0:
                    test_inputs, test_labels = get_next_batch(1)             
                    feed = {inputs: test_inputs, labels: test_labels, keep_prob: 1}
                    b_predictions = session.run([predictions], feed)                     
                    b_predictions = np.reshape(b_predictions[0],test_labels[0].shape)   
                    _pred = np.transpose(b_predictions)        
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_input.png"%steps), np.transpose(test_inputs[0]*255))
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_label.png"%steps), np.transpose(test_labels[0]*255))
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_pred.png"%steps), _pred*255)
                    # cv2.imwrite(os.path.join(curr_dir,"test","%s_input.png"%steps), np.transpose(test_inputs[0]))
                    # cv2.imwrite(os.path.join(curr_dir,"test","%s_label.png"%steps), np.transpose(test_labels[0]))
                    # cv2.imwrite(os.path.join(curr_dir,"test","%s_pred.png"%steps), _pred)

            saver.save(session, saver_prefix, global_step=steps)
                
if __name__ == '__main__':
    train()