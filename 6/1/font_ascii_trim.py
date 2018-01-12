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

# 参考 https://github.com/kaonashi-tyc/zi2zi/blob/master/model/unet.py
def TRIM_G(inputs, reuse=False):    
    with tf.variable_scope("TRIM_G", reuse=reuse):      
        layer, half_layer = utils_nn.pix2pix_g2(inputs)
        return layer, half_layer

def TRIM_D(inputs, reuse=False):
    with tf.variable_scope("TRIM_D", reuse=reuse):
        layer = utils_nn.pix2pix_d2(inputs)
        return layer

def CLEAN_G(inputs, reuse=False):    
    with tf.variable_scope("CLEAN_G", reuse=reuse):      
        layer, half_layer = utils_nn.pix2pix_g2(inputs)
        return layer, half_layer

def CLEAN_D(inputs, reuse=False):
    with tf.variable_scope("CLEAN_D", reuse=reuse):
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
    g_half_loss = tf.losses.mean_squared_error(half_real_A, half_real_B)   

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

# 降噪网络
def neural_networks_clean():
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size], name="inputs")
    targets = tf.placeholder(tf.float32, [None, image_size, image_size], name="targets")
    labels = tf.sparse_placeholder(tf.int32, name="labels")

    global_step = tf.Variable(0, trainable=False)
    real_A = tf.reshape(inputs, (-1, image_size, image_size, 1))
    real_B = tf.reshape(targets, (-1, image_size, image_size, 1))

    # 对抗网络
    fake_B, half_real_A = CLEAN_G(real_A, reuse = False)
    real_AB = tf.concat([real_A, real_B], 3)
    fake_AB = tf.concat([real_A, fake_B], 3)
    real_D  = CLEAN_D(real_AB, reuse = False)
    fake_D  = CLEAN_D(fake_AB, reuse = True)

    # 假设预计输出和真实输出应该在一半网络也应该是相同的
    _, half_real_B = CLEAN_G(fake_B, reuse = True)
    g_half_loss = tf.losses.mean_squared_error(half_real_A, half_real_B)   

    d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_D), real_D)
    d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_D), fake_D)
    d_loss  = d_loss_real + d_loss_fake

    g_loss_fake = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_D), fake_D)
    g_mse_loss = tf.losses.mean_squared_error(real_B, fake_B)

    g_loss     = g_loss_fake + g_mse_loss + g_half_loss
    
    g_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CLEAN_G')
    d_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CLEAN_D')

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
    clear_images = []
    half_clear_images = []
    max_width_image = 0
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
            text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)
            image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint)
            temp_image = utils_pil.resize_by_height(image, image_height)
            w, h = temp_image.size
            if w * h <= image_size * image_size: break
        image = utils_pil.convert_to_gray(image)    #原始图片   
        w, h = image.size
        if h > image_height:
            image = utils_pil.resize_by_height(image, image_height)     
        source_image = image.copy()

        # 随机缩放下图片
        if random.random()>0.5:
            _h =  random.randint(9, image_height+1)
            image = utils_pil.resize_by_height(image, _h)  
        # image = utils_pil.resize_by_height(image, image_height, random.random()>0.5) 

        # 干净的图片，给降噪网络用
        clears_image = source_image.copy()
        w, h = clears_image.size
        if h != image_height:
            clears_image = utils_pil.resize_by_height(clears_image, image_height)
        clears_image = np.asarray(clears_image)
        clears_image = (255. - clears_image) / 255. 
        clear_images.append(clears_image)
      
        # 给clear降噪网络使用
        half_clear_image = image.copy()
        w, h = half_clear_image.size
        if h != image_height:
            half_clear_image = utils_pil.resize_by_height(half_clear_image, image_height, random.random()>0.5)
        half_clear_image = utils_font.add_noise(half_clear_image) 
        half_clear_image = np.asarray(half_clear_image)
        half_clear_image = half_clear_image * random.uniform(0.3, 1)
        if random.random()>0.5:
            half_clear_image = (255. - half_clear_image) / 255.
        else:
            half_clear_image = half_clear_image / 255.           
        half_clear_images.append(half_clear_image) 

        # 随机移动位置 trims_image 为字体实际位置标识
        image = utils_pil.random_space2(image, image_height)
        trims_image = np.asarray(image)
        trims_image = (255. - trims_image) / 255.         
        trim_images.append(trims_image)

        if random.random()>0.1:
            image = utils_font.add_noise(image)   
        image = np.asarray(image)
        if random.random()>0.1:
            image = image * random.uniform(0.3, 1)
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

    clears = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        clears[i,:] = utils.square_img(clear_images[i], np.zeros([image_size, image_size]), image_height)

    half_clears = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        half_clears[i,:] = utils.square_img(half_clear_images[i], np.zeros([image_size, image_size]), image_height)

    return inputs, trims, clears, half_clears

t_d_saver, t_g_saver, c_d_saver, c_g_saver = [None, None, None, None]
t_model_D_dir, t_model_G_dir, c_model_D_dir, c_model_G_dir = [None, None, None, None]

def init_saver():
    global t_d_saver, t_g_saver, c_d_saver, c_g_saver, t_model_D_dir, t_model_G_dir, c_model_D_dir, c_model_G_dir  
    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    t_model_D_dir = os.path.join(model_dir, "TD")
    t_model_G_dir = os.path.join(model_dir, "TG")
    c_model_D_dir = os.path.join(model_dir, "CD")
    c_model_G_dir = os.path.join(model_dir, "CG")
    if not os.path.exists(t_model_D_dir): os.mkdir(t_model_D_dir)
    if not os.path.exists(t_model_G_dir): os.mkdir(t_model_G_dir)  
    if not os.path.exists(c_model_D_dir): os.mkdir(c_model_D_dir)
    if not os.path.exists(c_model_G_dir): os.mkdir(c_model_G_dir)  
    t_d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TRIM_D'), sharded=True)
    t_g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='TRIM_G'), sharded=True)
    c_d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CLEAN_D'), sharded=True)
    c_g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CLEAN_G'), sharded=True)

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

    if saver == None or saver == c_g_saver: 
        ckpt = tf.train.get_checkpoint_state(c_model_G_dir)
        if ckpt and ckpt.model_checkpoint_path:           
            print("Restore Model CLEAN G...")
            c_g_saver.restore(session, ckpt.model_checkpoint_path)   

    if saver == None or saver == c_d_saver: 
        ckpt = tf.train.get_checkpoint_state(c_model_D_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model CLEAN D...")
            c_d_saver.restore(session, ckpt.model_checkpoint_path) 

def save(session, saver, steps):
    if saver == None or saver == t_g_saver:
        print("Save Model TRIM G...")
        t_g_saver.save(session, os.path.join(t_model_G_dir, "TG.ckpt"), global_step=steps)
    if saver == None or saver == t_d_saver: 
        print("Save Model TRIM D...")
        t_d_saver.save(session, os.path.join(t_model_D_dir, "TD.ckpt"), global_step=steps)
    if saver == None or saver == c_g_saver: 
        print("Save Model CLEAN G...")
        c_g_saver.save(session, os.path.join(c_model_G_dir, "CG.ckpt"), global_step=steps)
    if saver == None or saver == c_d_saver: 
        print("Save Model CLEAN D...")
        c_d_saver.save(session, os.path.join(c_model_D_dir, "CD.ckpt"), global_step=steps)
  

def train():
    t_inputs, t_targets, t_global_step, \
        t_g_optim_mse, t_d_loss, t_d_loss_real, t_d_loss_fake, t_d_optim, \
        t_g_loss, t_g_mse_loss, t_g_half_loss, t_g_loss_fake, t_g_optim, t_fake_B = neural_networks_trim()

    c_inputs, c_targets, c_global_step, \
        c_g_optim_mse, c_d_loss, c_d_loss_real, c_d_loss_fake, c_d_optim, \
        c_g_loss, c_g_mse_loss, c_g_half_loss, c_g_loss_fake, c_g_optim, c_fake_B = neural_networks_clean()

    init_saver()
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        restore(session)

        while True:
            errA = errD1 = errD2 = 1
            for batch in range(BATCHES):
                batch_size = 16
                train_inputs, train_trims, train_clears, train_half_clears = get_next_batch_for_gan(batch_size)
                feed = {t_inputs: train_inputs, t_targets: train_trims}

                start = time.time()                
                errD, errD1, errD2, _, steps = session.run([t_d_loss, t_d_loss_real, t_d_loss_fake, t_d_optim, t_global_step], feed)
                print("T %d time: %4.4fs, d_loss: %.8f (d_loss_real: %.6f  d_loss_fake: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))
                
                start = time.time()                
                errD, errD1, errD2, _, steps = session.run([t_d_loss, t_d_loss_real, t_d_loss_fake, t_d_optim, t_global_step], feed)
                print("T %d time: %4.4fs, d_loss: %.8f (d_loss_real: %.6f  d_loss_fake: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))
                
                start = time.time()                
                errD, errD1, errD2, _, steps = session.run([t_d_loss, t_d_loss_real, t_d_loss_fake, t_d_optim, t_global_step], feed)
                print("T %d time: %4.4fs, d_loss: %.8f (d_loss_real: %.6f  d_loss_fake: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))

                start = time.time()                                
                errG, errM, errA, errH, _, steps, t_net_g = session.run([t_g_loss, t_g_mse_loss, t_g_loss_fake, t_g_half_loss, t_g_optim, t_global_step, t_fake_B], feed)
                print("T %d time: %4.4fs, g_loss: %.8f (mse: %.6f half: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errH, errA))

                # train_clean_inputs = np.zeros([batch_size, image_size, image_size])
                # has_err = False
                # for i in range(batch_size):
                #     _t_net_g = np.squeeze(t_net_g[i], axis=2)
                #     dstimg = utils.img2mask(train_inputs[i], _t_net_g, image_height, 0.3) 
                #     try:
                #         dstimg = utils.dropZeroEdges(dstimg) 
                #     except:
                #         cv2.imwrite(os.path.join(curr_dir,"test","E%s_%s_0.png"%(steps,i)), train_inputs[i] * 255) 
                #         cv2.imwrite(os.path.join(curr_dir,"test","E%s_%s_1.png"%(steps,i)), train_trims[i]) 
                #         cv2.imwrite(os.path.join(curr_dir,"test","E%s_%s_2.png"%(steps,i)), _t_net_g * 255) 
                #         cv2.imwrite(os.path.join(curr_dir,"test","E%s_%s_3.png"%(steps,i)), dstimg * 255)                            
                #     dstimg = utils.resize(dstimg, image_height)
                #     w,h = dstimg.shape
                #     if w * h > image_size * image_size :
                #         print("trim src image failed, break")
                #         has_err = True
                #         break
                #     train_clean_inputs[i,:] = utils.square_img(dstimg,np.zeros([image_size, image_size]))
                # if has_err: continue

                # feed = {c_inputs: train_half_clears, c_targets: train_clears}

                # start = time.time()                
                # errD, errD1, errD2, _, steps = session.run([c_d_loss, c_d_loss_real, c_d_loss_fake, c_d_optim, c_global_step], feed)
                # print("C %d time: %4.4fs, d_loss: %.8f (d_loss_real: %.6f  d_loss_fake: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))

                # start = time.time()                                
                # errG, errM, errA, errH, _, steps, c_net_g = session.run([c_g_loss, c_g_mse_loss, c_g_loss_fake, c_g_half_loss, c_g_optim, c_global_step, c_fake_B], feed)
                # print("C %d time: %4.4fs, g_loss: %.8f (mse: %.6f half: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errH, errA))

                # 报告
                if steps > 0 and steps % REPORT_STEPS < 4:
                    for i in range(batch_size): 
                        _t_net_g = np.squeeze(t_net_g[i], axis=2)
                        # _c_net_g = np.squeeze(c_net_g[i], axis=2)

                        _t_img = utils.unsquare_img(_t_net_g, image_height)                        
                        # _t_img_bin=np.where(_t_img>0.5, 1, 0)
                        _t_img_bin = np.copy(_t_img)    
                        _t_img_bin[_t_img_bin<=0.2] = 0
                        # _img = np.vstack((_t_img, _t_img_bin))                         
                        # cv2.imwrite(os.path.join(curr_dir,"test","S%s_%s.png"%(steps,i)), _img * 255)
                        _t_img = utils.dropZeroEdges(_t_img_bin, _t_img, min_rate=0.1)
                        _t_img = utils.resize(_t_img, image_height)
                        if _t_img.shape[0] * _t_img.shape[1] <= image_size * image_size:
                            _t_net_g = utils.square_img(_t_img, np.zeros([image_size, image_size]), image_height)

                        _img = np.vstack((train_inputs[i], _t_net_g)) 
                        cv2.imwrite(os.path.join(curr_dir,"test","T%s_%s.png"%(steps,i)), _img * 255) 
                        # _img = np.vstack((train_half_clears[i], _c_net_g)) 
                        # cv2.imwrite(os.path.join(curr_dir,"test","C%s_%s.png"%(steps,i)), _img * 255) 
            save(session, t_d_saver, t_global_step)
            save(session, t_g_saver, t_global_step)
            # save(session, c_d_saver, c_global_step)
            # save(session, c_g_saver, c_global_step)
            
if __name__ == '__main__':
    train()