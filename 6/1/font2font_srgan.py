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
CLASSES_NUMBER = len(CHARS)

#初始化学习速率
LEARNING_RATE_INITIAL = 1e-3
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 200
MOMENTUM = 0.9

BATCHES = 64
BATCH_SIZE = 8
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
        for n in (1,2,4,8,16,32,16,8):
            layer = slim.conv2d(layer, df_dim * n, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
            layer = slim.batch_norm(layer, activation_fn = tf.nn.relu)
        net = layer
        for n in (2,2,8):
            net = slim.conv2d(net, df_dim * n, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
            net = slim.batch_norm(net, activation_fn = tf.nn.relu)            
        net = tf.nn.relu(net + layer)
        net = tf.reshape(net, [-1, df_dim * 8])
        logits = slim.fully_connected(net, 1, activation_fn=tf.identity)
        net_ho = tf.nn.sigmoid(logits)
        return net_ho, logits

def vgg19(inputs, reuse = False):
    with tf.variable_scope("VGG19", reuse=reuse):
        layer = slim.conv2d(inputs, 64, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 64, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.max_pool2d(layer, [2, 2], padding="SAME", stride=2)
        layer = slim.conv2d(layer, 128, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 128, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.max_pool2d(layer, [2, 2], padding="SAME", stride=2)
        layer = slim.conv2d(layer, 256, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 256, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 256, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 256, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.max_pool2d(layer, [2, 2], padding="SAME", stride=2)
        layer = slim.conv2d(layer, 512, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 512, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 512, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 512, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.max_pool2d(layer, [2, 2], padding="SAME", stride=2)
        conv = layer
        layer = slim.conv2d(layer, 512, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 512, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 512, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.conv2d(layer, 512, [3,3], normalizer_fn = None, activation_fn = tf.nn.relu)
        layer = slim.max_pool2d(layer, [2, 2], padding="SAME", stride=2)
        layer = tf.reshape(layer, [-1,512])
        layer = slim.fully_connected(layer, 4096, activation_fn=tf.nn.relu)   
        layer = slim.fully_connected(layer, 4096, activation_fn=tf.nn.relu)   
        layer = slim.fully_connected(layer, 1000, activation_fn=tf.nn.relu)    
        
        shape = tf.shape(inputs)
        batch_size, image_width = shape[0], shape[1]        
        layer = tf.reshape(layer,[batch_size, -1, 1000]) 

        layer = tf.transpose(layer, (0, 2, 1)) 
        cell_fw = tf.contrib.rnn.GRUCell(8)
        cell_bw = tf.contrib.rnn.GRUCell(8)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, layer, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=2)
        layer = tf.reshape(outputs, [-1, 1000*16])    
        layer = slim.fully_connected(layer, CLASSES_NUMBER , activation_fn=tf.identity)    
   
        return layer, conv

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, None, image_height], name="inputs")
    targets = tf.placeholder(tf.float32, [None, None, image_height], name="targets")
    labels = tf.placeholder(tf.int32, [None], name="labels") 
    global_step = tf.Variable(0, trainable=False)

    shape = tf.shape(inputs)
    batch_size, image_width = shape[0], shape[1]

    layer = tf.reshape(inputs, (batch_size, image_width, image_height, 1))
    layer_targets = tf.reshape(targets, (batch_size, image_width, image_height, 1))

    net_vgg, _ = vgg19(layer, reuse = False)
    vgg_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VGG19')
    vgg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net_vgg, labels=labels)
    vgg_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(vgg_loss, global_step=global_step, var_list=vgg_vars)

    net_g = SRGAN_g(layer, reuse = False)
    net_d, logits_real = SRGAN_d(layer_targets, reuse = False)
    _,     logits_fake = SRGAN_d(net_g, reuse = True)

    _, vgg_target_emb   = vgg19(layer_targets, reuse = True)
    _, vgg_predict_emb  = vgg19(net_g, reuse = True)

    d_loss1 = tf.losses.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
    d_loss2 = tf.losses.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
    d_loss  = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tf.losses.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_real))
    g_mse_loss   = tf.losses.mean_squared_error(net_g, layer_targets)
    g_vgg_loss   = 2e-6 * tf.losses.mean_squared_error(vgg_target_emb, vgg_predict_emb)
    g_loss     = g_gan_loss + g_mse_loss + g_vgg_loss
    
    g_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g')
    d_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_d')

    g_optim_init = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_mse_loss, global_step=global_step, var_list=g_vars)

    g_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_loss, global_step=global_step, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(d_loss, global_step=global_step, var_list=d_vars)

    return inputs, targets, labels, global_step, g_optim_init, d_loss, d_optim, \
            g_loss, g_mse_loss, g_vgg_loss, g_gan_loss, g_optim, net_g, vgg_loss, vgg_optim


ENGFontNames, CHIFontNames = utils_font.get_font_names_from_url()
print("EngFontNames", ENGFontNames)
print("CHIFontNames", CHIFontNames)
AllFontNames = ENGFontNames + CHIFontNames

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 

def get_vgg_next_batch(batch_size=128):
    images = []  
    labels = []
    max_width_image = 0
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        # font_length = random.randint(font_min_length-5, font_min_length+5)        
        font_size = random.randint(image_height, 64)    
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,2,3,4,5]) 
        text  = utils_font.get_random_text(CHARS, eng_world_list, 1)[0]    
        image = utils_font.get_font_image_from_url(text, font_name ,font_size)
        image = utils_pil.convert_to_gray(image)
        image = np.asarray(image)     
        image = utils.resize(image, height=image_height)
        image = (255. - image) / 255.
        images.append(image)
        labels.append(CHARS.index(text))
        if image.shape[1] > max_width_image: 
            max_width_image = image.shape[1]
            
    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = utils.img2vec(images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.transpose(image_vec)                   
    return inputs, labels
    
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
    
    inputs, targets, labels, global_step, g_optim_init, d_loss, d_optim, \
        g_loss, g_mse_loss, g_vgg_loss, g_gan_loss, g_optim, net_g, vgg_loss, vgg_optim = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    saver_prefix = os.path.join(model_dir, "model.ckpt")        
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.Saver(max_to_keep=5)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model ...")
            saver.restore(session, ckpt.model_checkpoint_path)    

        steps = session.run(global_step)
        # train vgg19
        while steps < 100000:
            for batch in range(BATCHES):
                start = time.time() 
                train_inputs, train_labels = get_vgg_next_batch(BATCH_SIZE)
                errM, _ , steps= session.run([vgg_loss, vgg_optim, global_step], {inputs: train_inputs, labels: train_labels})
                print("%8d time: %4.4fs, mse: %.8f " % (steps, time.time() - start, errM))
            saver.save(session, saver_prefix, global_step=steps)                

        # initialize G
        while steps < 200000:
            for batch in range(BATCHES):
                start = time.time() 
                train_inputs, train_labels = get_next_batch(BATCH_SIZE)
                errM, _ , steps= session.run([g_mse_loss, g_optim_init, global_step], {inputs: train_inputs, targets: train_labels})
                print("%8d time: %4.4fs, mse: %.8f " % (steps, time.time() - start, errM))
            saver.save(session, saver_prefix, global_step=steps)

        # train GAN (SRGAN)
        while True:
            for batch in range(BATCHES):
                start = time.time()                
                train_inputs, train_labels = get_next_batch(BATCH_SIZE)  

                ## update D
                errD, _ = session.run([d_loss, d_optim], {inputs: train_inputs, targets: train_labels})
                ## update G
                errG, errM, errV, errA, _, steps = session.run([g_loss, g_mse_loss, g_vgg_loss, g_gan_loss, g_optim, global_step],
                     {inputs: train_inputs, targets: train_labels})
                print("%8d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" % (steps, time.time() - start, errD, errG, errM, errV, errA))

                if np.isnan(errG) or np.isinf(errG) or np.isnan(errA) or np.isinf(errA) or np.isnan(errD) or np.isinf(errD):
                    print("Error: cost is nan or inf")
                    return   
                
                if time.time() - start > 60: 
                    print('Exit for long time')
                    return

                if steps > 0 and steps % REPORT_STEPS == 0:
                    test_inputs, test_labels = get_next_batch(1)             
                    feed = {inputs: test_inputs, targets: test_labels}
                    b_predictions = session.run([net_g], feed)                     
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