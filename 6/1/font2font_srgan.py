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
CLASSES_NUMBER = len(CHARS) + 2 

#初始化学习速率
LEARNING_RATE_INITIAL = 1e-3
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 200
MOMENTUM = 0.9

BATCHES = 64
BATCH_SIZE = 6
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_font2font_srgan"

# 增加残差网络
def addResLayer(inputs):
    layer = slim.batch_norm(inputs, activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.conv2d(layer, 64, [3,3], activation_fn=None)
    layer = slim.batch_norm(layer, activation_fn=None)
    layer = tf.nn.relu(layer)
    layer = slim.conv2d(layer, 64, [3,3],activation_fn=None)
    outputs = inputs + layer
    return outputs 

# 增加 Highway 网络
def addHighwayLayer(inputs):
    H = slim.conv2d(inputs, 64, [3,3])
    T = slim.conv2d(inputs, 64, [3,3], biases_initializer = tf.constant_initializer(-1.0), activation_fn=tf.nn.sigmoid)    
    outputs = H * T + inputs * (1.0 - T)
    return outputs    

def SRGAN_g(inputs, reuse=False):    
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        layer = slim.conv2d(inputs, 64, [3,3], normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
        temp = layer
        # B residual blocks
        for i in range(16):
            layer = addResLayer(layer)
        layer = slim.conv2d(layer, 64, [3,3], normalizer_fn = None, activation_fn = None)
        layer = slim.batch_norm(layer, activation_fn=None)
        layer = layer + temp        
        # B residual blacks end
        layer = slim.conv2d(layer, 256, [3,3], activation_fn=tf.nn.relu)
        layer = slim.conv2d(layer, 256, [3,3], activation_fn=tf.nn.relu)
        layer = slim.conv2d(layer, 1,   [1,1], activation_fn=tf.nn.tanh)
        return layer

def SRGAN_d(inputs, reuse=False):
    df_dim = 64
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        layer = inputs
        for n in (1,2,4,8,16,8):
            layer = slim.conv2d(layer, df_dim * n, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
            layer = slim.batch_norm(layer, activation_fn = None)
        net = layer
        for n in (1,2,4,8):
            net = slim.conv2d(net, df_dim * n, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
            net = slim.batch_norm(net, activation_fn = None)            
        net = tf.nn.relu(net + layer)
        logits = slim.fully_connected(net, 1, activation_fn=tf.identity)
        net_ho = tf.nn.sigmoid(logits)
        return net_ho, logits

def Highway(inputs, reuse = False):
    with tf.variable_scope("HIGHWAY", reuse=reuse):
        layer = slim.conv2d(inputs, 64, [3,3], normalizer_fn=slim.batch_norm)
        for i in range(POOL_COUNT):
            for j in range(16):
                layer = addHighwayLayer(layer)
            layer = slim.conv2d(layer, 64, [3,3], stride=[2, 2], normalizer_fn=slim.batch_norm)  
        conv = layer
        layer = slim.conv2d(layer, CLASSES_NUMBER, [3,3], normalizer_fn=slim.batch_norm, activation_fn=None)
        shape = tf.shape(inputs)
        batch_size, image_width = shape[0], shape[1]        
        layer = tf.reshape(layer, [batch_size, -1, CLASSES_NUMBER])
        layer = tf.transpose(layer, (1, 0, 2))       
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
    net_d, logits_real = SRGAN_d(layer_targets, reuse = False)
    _,     logits_fake = SRGAN_d(net_g, reuse = True)

    net_highway, _ = Highway(net_g, reuse = False)
    seq_len = tf.placeholder(tf.int32, [None])
    highway_vars  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='HIGHWAY')
    highway_loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=net_highway, sequence_length=seq_len))
    highway_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(highway_loss, global_step=global_step, var_list=highway_vars)
    highway_decoded, _ = tf.nn.ctc_beam_search_decoder(net_highway, seq_len, beam_width=10, merge_repeated=False)
    highway_acc = tf.reduce_mean(tf.edit_distance(tf.cast(highway_decoded[0], tf.int32), labels))

    _, highway_target_emb   = Highway(layer_targets, reuse = True)
    _, highway_predict_emb  = Highway(net_g, reuse = True)

    d_loss1 = 1e-3*tf.losses.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
    d_loss2 = 1e-3*tf.losses.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
    d_loss  = d_loss1 - d_loss2

    g_gan_loss = 1e-3*tf.losses.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
    g_mse_loss = tf.losses.mean_squared_error(net_g, layer_targets)
    g_highway_loss = tf.losses.mean_squared_error(highway_target_emb, highway_predict_emb)
    g_loss     = g_gan_loss + g_mse_loss + g_highway_loss
    
    g_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g')
    d_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_d')

    g_optim_init = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_mse_loss, global_step=global_step, var_list=g_vars)

    g_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_loss, global_step=global_step, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(d_loss, global_step=global_step, var_list=d_vars)

    return inputs, targets, labels, global_step, g_optim_init, d_loss, d_loss1, d_loss2, d_optim, \
            g_loss, g_mse_loss, g_highway_loss, g_gan_loss, g_optim, net_g, highway_loss, highway_optim, seq_len, highway_acc


ENGFontNames, CHIFontNames = utils_font.get_font_names_from_url()
print("EngFontNames", ENGFontNames)
print("CHIFontNames", CHIFontNames)
AllFontNames = ENGFontNames + CHIFontNames

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 

# 转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape

# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch(batch_size=128):
    images = []   
    to_images = []
    codes = []
    max_width_image = 0
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        font_length = random.randint(13, 15)
        font_size = random.randint(image_height, 64)    
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,2,3,4,5])     
        text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)
        codes.append([CHARS.index(char) for char in text])          
        image = utils_font.get_font_image_from_url(text, font_name, font_size, fontmode = font_mode, fonthint = font_hint )
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

    targets = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(to_images)):
        image_vec = utils.img2vec(to_images[i], height=image_height, width=max_width_image, flatten=False)
        targets[i,:] = np.transpose(image_vec)

    labels = [np.asarray(i) for i in codes]
    sparse_labels = sparse_tuple_from(labels)
    seq_len = np.ones(batch_size) * (max_width_image * image_height ) // (POOL_SIZE * POOL_SIZE)                
    return inputs, targets, sparse_labels, seq_len

def train():
    inputs, targets, labels, global_step, g_optim_init, d_loss, d_loss1, d_loss2, d_optim, \
        g_loss, g_mse_loss, g_highway_loss, g_gan_loss, g_optim, net_g, \
        highway_loss, highway_optim, seq_len, highway_acc = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_H_dir = os.path.join(model_dir, "H")
    model_D_dir = os.path.join(model_dir, "D")
    model_G_dir = os.path.join(model_dir, "G")
    if not os.path.exists(model_H_dir): os.mkdir(model_H_dir)
    if not os.path.exists(model_D_dir): os.mkdir(model_D_dir)
    if not os.path.exists(model_G_dir): os.mkdir(model_G_dir)
    

    # saver_prefix = os.path.join(model_dir, "model.ckpt")        
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        
        h_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='HIGHWAY'), max_to_keep=5)
        d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_d'), max_to_keep=5)
        g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g'), max_to_keep=5)

        ckpt = tf.train.get_checkpoint_state(model_G_dir)
        if ckpt and ckpt.model_checkpoint_path:           
            print("Restore Model G...")
            g_saver.restore(session, ckpt.model_checkpoint_path)   
        ckpt = tf.train.get_checkpoint_state(model_H_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model H...")
            h_saver.restore(session, ckpt.model_checkpoint_path)   
        ckpt = tf.train.get_checkpoint_state(model_D_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model D...")
            d_saver.restore(session, ckpt.model_checkpoint_path)    
 

        while True:
            for batch in range(BATCHES):
                for i in range(3):
                    train_inputs, train_targets, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)
                    feed = {inputs: train_inputs, targets: train_targets, labels: train_labels, seq_len: train_seq_len}
                
                    # # train G
                    # start = time.time() 
                    # errM, _ , steps= session.run([g_mse_loss, g_optim_init, global_step], feed)
                    # print("%8d time: %4.4fs, g_mse_loss: %.8f " % (steps, time.time() - start, errM))
                    # if np.isnan(errM) or np.isinf(errM) :
                    #     print("Error: cost is nan or inf")
                    #     return                    

                    # train highway
                    start = time.time() 
                    errM, acc, _ , steps= session.run([highway_loss, highway_acc, highway_optim, global_step], feed)
                    print("%d time: %4.4fs, highway_loss: %.8f, highway_acc: %.8f " % (steps, time.time() - start, errM, acc))
                    if np.isnan(errM) or np.isinf(errM) :
                        print("Error: cost is nan or inf")
                        return   
  
                    start = time.time()                                
                    ## update G
                    errG, errM, errV, errA, _, steps = session.run([g_loss, g_mse_loss, g_highway_loss, g_gan_loss, g_optim, global_step], feed)
                    print("%d time: %4.4fs, g_loss: %.8f (mse: %.6f highway: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errV, errA))
                    if np.isnan(errG) or np.isinf(errG) or np.isnan(errA) or np.isinf(errA):
                        print("Error: cost is nan or inf")
                        return 

                train_inputs, train_targets, train_labels, train_seq_len = get_next_batch(BATCH_SIZE)
                feed = {inputs: train_inputs, targets: train_targets, labels: train_labels, seq_len: train_seq_len}

                # train GAN (SRGAN)
                start = time.time()                
                ## update D
                errD, errD1, errD2, _, steps = session.run([d_loss, d_loss1, d_loss2, d_optim, global_step], feed)
                print("%d time: %4.4fs, d_loss: %.8f (d_loss1: %.6f  d_loss2: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))
                if np.isnan(errD) or np.isinf(errD):
                    print("Error: cost is nan or inf")
                    return 
                
                if steps > 0 and steps % REPORT_STEPS < 7:
                    train_inputs, train_targets, train_labels, train_seq_len = get_next_batch(1)             
                    feed = {inputs: train_inputs, targets: train_targets}
                    b_predictions = session.run([net_g], feed)                     
                    b_predictions = np.reshape(b_predictions[0],train_targets[0].shape)   
                    _pred = np.transpose(b_predictions)        
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_input.png"%steps), np.transpose(train_inputs[0]*255))
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_label.png"%steps), np.transpose(train_targets[0]*255))
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_pred.png"%steps), _pred*255)

            print("save model ...")
            h_saver.save(session, os.path.join(model_H_dir, "H.ckpt"), global_step=steps)
            d_saver.save(session, os.path.join(model_D_dir, "D.ckpt"), global_step=steps)
            g_saver.save(session, os.path.join(model_G_dir, "G.ckpt"), global_step=steps)
                
if __name__ == '__main__':
    train()