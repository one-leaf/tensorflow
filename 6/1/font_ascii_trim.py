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
LEARNING_RATE_INITIAL = 1e-4
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 500
MOMENTUM = 0.9

BATCHES = 32
BATCH_SIZE = 4
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_ascii"

# 参考 https://github.com/kaonashi-tyc/zi2zi/blob/master/model/unet.py
def TRIM_G(inputs, reuse=False):    
    with tf.variable_scope("TRIM_G", reuse=reuse):      
        layer, half_layer = utils_nn.pix2pix_g3(inputs)
        return layer, half_layer

def TRIM_D(inputs, reuse=False):
    with tf.variable_scope("TRIM_D", reuse=reuse):
        layer = utils_nn.pix2pix_d2(inputs)
        return layer

# 位置调整
def neural_networks_trim():
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size], name="inputs")
    targets = tf.placeholder(tf.float32, [None, image_size, image_size], name="targets")

    global_step = tf.Variable(0, trainable=False)
    real_A = tf.reshape(inputs, (-1, image_size, image_size, 1))
    real_B = tf.reshape(targets, (-1, image_size, image_size, 1))

    # 对抗网络
    fake_B, half_real_A = TRIM_G(real_A, reuse = False)
    real_AB = tf.concat([real_A, real_B], 3)
    fake_AB = tf.concat([real_A, fake_B], 3)
    real_D  = TRIM_D(real_AB, reuse = False)
    fake_D  = TRIM_D(fake_AB, reuse = True)

    # 假设预计输出和真实输出应该在一半网络也应该是相同的
    _, half_real_B = TRIM_G(fake_B, reuse = True)
    g_half_loss = 1e3 * tf.losses.mean_squared_error(half_real_A, half_real_B)   

    d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_D), real_D)
    d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_D), fake_D)
    d_loss  = d_loss_real + d_loss_fake

    g_loss_fake = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_D), fake_D)
    g_mse_loss = tf.losses.mean_squared_error(real_B, fake_B)

    g_loss     = g_loss_fake + g_mse_loss + g_half_loss
    
    g_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TRIM_G')
    d_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TRIM_D')

    g_optim_mse = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_mse_loss, global_step=global_step, var_list=g_vars)
    g_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_loss, global_step=global_step, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(d_loss, global_step=global_step, var_list=d_vars)

    return  inputs, targets, global_step, \
            g_optim_mse, d_loss, d_loss_real, d_loss_fake, d_optim, \
            g_loss, g_mse_loss, g_half_loss, g_loss_fake, g_optim, fake_B



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
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        # font_size = image_height #random.randint(image_height, 64)    
        # if font_size==None:
        if random.random()>0.5:
            font_size = random.randint(9, 49)    
        else:
            font_size = random.randint(9, 15) 
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,2,3,4,5])     #删除了2
        while True:
            font_length = random.randint(3, 400)            
            # text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)
            text  = utils_font.get_words_text(CHARS, eng_world_list, font_length)
            image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint, trim=False)
            if font_hint in (0,1,3,5):
                clear_trim_image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, 0, trim=False)
            else:
                clear_trim_image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, 4, trim=False)   

            bbox = utils_pil.get_trim_box(clear_trim_image)
            image = image.crop(bbox)
            clear_trim_image = clear_trim_image.crop(bbox)

            temp_image = utils_pil.resize_by_height(image, image_height)
            if clear_trim_image.size[0] != image.size[0] or  clear_trim_image.size[1] != image.size[1]:
                print("get size not same,",image.size,clear_trim_image.size,font_name,font_size,font_mode,font_hint)
                continue
            w, h = temp_image.size
            if w * h <= image_size * image_size: 
                break

        image = utils_pil.convert_to_gray(image)    #原始图片   
        clear_trim_image = utils_pil.convert_to_gray(clear_trim_image)

        w, h = image.size
        if h > image_height:
            image = utils_pil.resize_by_height(image, image_height)   
            clear_trim_image = utils_pil.resize_by_size(clear_trim_image, image.size)  
        source_image = image.copy()

        # 随机缩放下图片
        w, h = image.size
        if random.random()>0.5 and h<image_height:
            _h =  random.randint(h+1, image_height+1)
            image = utils_pil.resize_by_height(image, _h, random.random()>0.5)  
            clear_trim_image = utils_pil.resize_by_size(clear_trim_image, image.size)  
            if clear_trim_image.size[0] != image.size[0] or  clear_trim_image.size[1] != image.size[1]:
                print("random resize get size not same,",image.size,clear_trim_image.size,font_name,font_size,font_mode,font_hint)
        # image = utils_pil.resize_by_height(image, image_height, random.random()>0.5) 

             # 随机移动位置 trims_image 为字体实际位置标识
        image, clear_trim_image = utils_pil.random_space2(image, clear_trim_image, image_height)

        if clear_trim_image.size[0] != image.size[0] or  clear_trim_image.size[1] != image.size[1]:
            print("random move space get size not same,",image.size,clear_trim_image.size,font_name,font_size,font_mode,font_hint)

        trims_image = np.asarray(clear_trim_image)
        # 转黑白二值化，降低维度
        trims_image = (255. - trims_image) / 255.  
        trim_images.append(trims_image)

        if random.random()>0.9:
            image = utils_font.add_noise(image)   
        image = np.asarray(image)

        if random.random()>0.9:
            image[image>200] = image * random.uniform(0.5, 1)

        if random.random()>0.5:
            image = (255. - image) / 255.
        else:
            image = image / 255.           
        input_images.append(image)   

    inputs = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        inputs[i,:] = utils.square_img(input_images[i], np.zeros([image_size, image_size]), image_height)

    trims = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        trims[i,:] = utils.square_img(trim_images[i], np.zeros([image_size, image_size]), image_height)
        trims[trims==0] == -1

    return inputs, trims

t_d_saver, t_g_saver = [None, None]
t_model_D_dir, t_model_G_dir = [None, None]

def init_saver():
    global t_model_D_dir, t_model_G_dir, t_d_saver, t_g_saver  
    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    t_model_D_dir = os.path.join(model_dir, "TD32")
    t_model_G_dir = os.path.join(model_dir, "TG32")
    if not os.path.exists(t_model_D_dir): os.mkdir(t_model_D_dir)
    if not os.path.exists(t_model_G_dir): os.mkdir(t_model_G_dir)  
    t_d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TRIM_D'), sharded=True)
    t_g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TRIM_G'), sharded=True)

def restore(session, saver=None):
    if saver == None or saver == t_g_saver: 
        ckpt = tf.train.get_checkpoint_state(t_model_G_dir)
        if ckpt and ckpt.model_checkpoint_path:           
            print("Restore Model TRIM G...")
            t_g_saver.restore(session, ckpt.model_checkpoint_path)   

    if saver == None or saver == t_d_saver: 
        ckpt = tf.train.get_checkpoint_state(t_model_D_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model TRIM D...")
            t_d_saver.restore(session, ckpt.model_checkpoint_path)    


def save(session, saver, steps):
    if saver == None or saver == t_g_saver:
        print("Save Model TRIM G...")
        t_g_saver.save(session, os.path.join(t_model_G_dir, "TG.ckpt"), global_step=steps)
    if saver == None or saver == t_d_saver: 
        print("Save Model TRIM D...")
        t_d_saver.save(session, os.path.join(t_model_D_dir, "TD.ckpt"), global_step=steps)


def train():
    t_inputs, t_targets, t_global_step, \
        t_g_optim_mse, t_d_loss, t_d_loss_real, t_d_loss_fake, t_d_optim, \
        t_g_loss, t_g_mse_loss, t_g_half_loss, t_g_loss_fake, t_g_optim, t_fake_B = neural_networks_trim()

    init_saver()
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        restore(session)

        while True:
            errA = errD1 = errD2 = 1
            for batch in range(BATCHES):
                batch_size = 16
                train_inputs, train_trims = get_next_batch_for_gan(batch_size)
                feed = {t_inputs: train_inputs, t_targets: train_trims}

                start = time.time()                
                errD, errD1, errD2, _, steps = session.run([t_d_loss, t_d_loss_real, t_d_loss_fake, t_d_optim, t_global_step], feed)
                print("T %d time: %4.4fs, d_loss: %.8f (d_loss_real: %.6f  d_loss_fake: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))
                
                # start = time.time()                
                # errD, errD1, errD2, _, steps = session.run([t_d_loss, t_d_loss_real, t_d_loss_fake, t_d_optim, t_global_step], feed)
                # print("T %d time: %4.4fs, d_loss: %.8f (d_loss_real: %.6f  d_loss_fake: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))
                
                # start = time.time()                
                # errD, errD1, errD2, _, steps = session.run([t_d_loss, t_d_loss_real, t_d_loss_fake, t_d_optim, t_global_step], feed)
                # print("T %d time: %4.4fs, d_loss: %.8f (d_loss_real: %.6f  d_loss_fake: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))

                start = time.time()                                
                errG, errM, errA, errH, _, steps, t_net_g = session.run([t_g_loss, t_g_mse_loss, t_g_loss_fake, t_g_half_loss, t_g_optim, t_global_step, t_fake_B], feed)
                print("T %d time: %4.4fs, g_loss: %.8f (mse: %.6f half: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errH, errA))

                # start = time.time()                                
                # errG, errM, errA, errH, _, steps, t_net_g = session.run([t_g_loss, t_g_mse_loss, t_g_loss_fake, t_g_half_loss, t_g_optim, t_global_step, t_fake_B], feed)
                # print("T %d time: %4.4fs, g_loss: %.8f (mse: %.6f half: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errH, errA))

                # start = time.time()                                
                # errG, errM, errA, errH, _, steps, t_net_g = session.run([t_g_loss, t_g_mse_loss, t_g_loss_fake, t_g_half_loss, t_g_optim, t_global_step, t_fake_B], feed)
                # print("T %d time: %4.4fs, g_loss: %.8f (mse: %.6f half: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errH, errA))                
                # 报告
                if steps > 0 and steps % REPORT_STEPS < 5:
                    for i in range(batch_size): 
                        _t_net_g = np.squeeze(t_net_g[i], axis=2)
                        # _c_net_g = np.squeeze(c_net_g[i], axis=2)

                        _t_img = utils.unsquare_img(_t_net_g, image_height)   
                        _t_img[_t_img<0] = 0
                        _t_img = utils.cvTrimImage(_t_img)
                        _t_img = utils.resize(_t_img, image_height)
                        if _t_img.shape[0] * _t_img.shape[1] <= image_size * image_size:
                            _t_net_g = utils.square_img(_t_img, np.zeros([image_size, image_size]), image_height)

                        _tran_img = train_trims[i]
                        _tran_img[_tran_img<0] = 0

                        _img = np.vstack((train_inputs[i], _t_net_g)) 
                        cv2.imwrite(os.path.join(curr_dir,"test","T%s_%s.png"%(steps,i)), _img * 255) 
                        # _img = np.vstack((train_half_clears[i], _c_net_g)) 
                        # cv2.imwrite(os.path.join(curr_dir,"test","C%s_%s.png"%(steps,i)), _img * 255) 
            save(session, t_d_saver, t_global_step)
            save(session, t_g_saver, t_global_step)
            
if __name__ == '__main__':
    train()