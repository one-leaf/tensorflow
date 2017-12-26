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

BATCHES = 256
BATCH_SIZE = 4
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))
MODEL_SAVE_NAME = "model_ascii_srgan"

# 参考 https://github.com/kaonashi-tyc/zi2zi/blob/master/model/unet.py
def SRGAN_g(inputs, reuse=False):    
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:      
        layer, half_layer = utils_nn.pix2pix_g2(inputs)
        return layer, half_layer

def SRGAN_d(inputs, reuse=False):
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        layer = utils_nn.pix2pix_d2(inputs)
        return layer

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, image_size, image_size], name="inputs")
    # 干净的图片
    targets = tf.placeholder(tf.float32, [None, image_size, image_size], name="targets")
    labels = tf.sparse_placeholder(tf.int32, name="labels")
    global_step = tf.Variable(0, trainable=False)

    real_A = tf.reshape(inputs, (-1, image_size, image_size, 1))
    real_B = tf.reshape(targets, (-1, image_size, image_size, 1))

    # 对抗网络
    fake_B, half_real_A = SRGAN_g(real_A, reuse = False)
    real_AB = tf.concat([real_A, real_B], 3)
    fake_AB = tf.concat([real_A, fake_B], 3)
    real_D  = SRGAN_d(real_AB, reuse = False)
    fake_D  = SRGAN_d(fake_AB, reuse = True)

    # 假设预计输出和真实输出应该在一半网络也应该是相同的
    _, half_real_B = SRGAN_g(fake_B, reuse = True)
    g_half_loss = tf.losses.mean_squared_error(half_real_A, half_real_B)   

    d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_D), real_D)
    d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_D), fake_D)
    d_loss  = d_loss_real + d_loss_fake

    g_loss_fake = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_D), fake_D)
    g_mse_loss = tf.losses.mean_squared_error(real_B, fake_B)

    g_loss     = g_loss_fake + g_mse_loss + g_half_loss
    
    g_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g')
    d_vars     = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_d')

    g_optim_mse = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_mse_loss, global_step=global_step, var_list=g_vars)
    g_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL).minimize(g_loss, global_step=global_step, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(LEARNING_RATE_INITIAL*0.1).minimize(d_loss, global_step=global_step, var_list=d_vars)

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

def list_to_chars(list):
    return "".join([CHARS[v] for v in list])

# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch_for_srgan(batch_size=128):
    inputs_images  = []
    targets_images = []
    max_width_image = 0
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        font_length = random.randint(15, 20)
        font_size = 36 #random.randint(image_height, 64)    
        font_mode = random.choice([0,1,2,4]) 
        font_hint = random.choice([0,1,2,3,4,5])     #删除了2
        text  = utils_font.get_random_text(CHARS, eng_world_list, font_length)
        image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint)
        image = utils_pil.resize_by_height(image, image_height)
        image = utils_pil.convert_to_gray(image)
        targets_image = image.copy()

        _h =  random.randint(9, image_height // random.choice([1,1.5,2,2.5]))
        image = utils_pil.resize_by_height(image, _h)        
        image = utils_pil.resize_by_height(image, image_height, random.random()>0.5) 

        targets_image = np.asarray(targets_image)
        targets_image = (255. - targets_image) / 255. 
        # targets_image = np.reshape(targets_image,[-1])
        # targets_image = np.pad(targets_image,(0, image_size*image_size-np.size(targets_image)),"constant")
        # targets_image = np.reshape(targets_image, [image_size,image_size])  
        targets_images.append(targets_image)

        image = utils_font.add_noise(image)   
        image = np.asarray(image)
        # image = utils.resize(image, height=image_height)
        image = image * random.uniform(0.3, 1)
        if random.random()>0.5:
            image = (255. - image) / 255.
        else:
            image = image / 255.
        # image = np.reshape(image,[-1])
        # image = np.pad(image,(0, image_size*image_size-np.size(image)),"constant")
        # image = np.reshape(image, [image_size,image_size])            
        inputs_images.append(image)   

    inputs = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        targets[i,:] = utils.img2img(inputs[i],np.zeros([image_size, image_size])


        # inputs[i,:] = inputs_images[i]

    targets = np.zeros([batch_size, image_size, image_size])
    for i in range(batch_size):
        targets[i,:] = utils.img2img(targets_images[i],np.zeros([image_size, image_size])

    return inputs, targets

def train():
    inputs, targets, global_step, \
        g_optim_mse, d_loss, d_loss_real, d_loss_fake, d_optim, \
        g_loss, g_mse_loss, g_half_loss, g_loss_fake, g_optim, fake_B = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, MODEL_SAVE_NAME)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_D_dir = os.path.join(model_dir, "FD")
    model_G_dir = os.path.join(model_dir, "FG")
    if not os.path.exists(model_D_dir): os.mkdir(model_D_dir)
    if not os.path.exists(model_G_dir): os.mkdir(model_G_dir)  
 
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        d_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_d'), sharded=True)
        g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g'), sharded=True)

        ckpt = tf.train.get_checkpoint_state(model_G_dir)
        if ckpt and ckpt.model_checkpoint_path:           
            print("Restore Model G...")
            g_saver.restore(session, ckpt.model_checkpoint_path)   

        ckpt = tf.train.get_checkpoint_state(model_D_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model D...")
            d_saver.restore(session, ckpt.model_checkpoint_path)    

        while True:
            errA = errD2 = 1
            for batch in range(BATCHES):
                train_inputs, train_targets = get_next_batch_for_srgan(16)
                feed = {inputs: train_inputs, targets: train_targets}

                # start = time.time() 
                # errM, _ , steps= session.run([g_mse_loss, g_optim_mse, global_step], feed)
                # print("%d time: %4.4fs, g_mse_loss: %.8f " % (steps, time.time() - start, errM))
                # if np.isnan(errM) or np.isinf(errM) :
                #     print("Error: cost is nan or inf")
                #     return    

                # train GAN (SRGAN)
                # print("errA:", errA, errA > 0.65)
                # if errA >= 0.5 or errA >= errD2:
                    # update G
                start = time.time()                                
                errG, errM, errA, errH, _, steps = session.run([g_loss, g_mse_loss, g_loss_fake, g_half_loss, g_optim, global_step], feed)
                print("%d time: %4.4fs, g_loss: %.8f (mse: %.6f half: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errH, errA))

                # print("errD2:", errD2, errD2 > 0.65)
                # if errA < 0.5 or errD2 >= 0.5:
                start = time.time()                
                ## update D
                errD, errD1, errD2, _, steps = session.run([d_loss, d_loss_real, d_loss_fake, d_optim, global_step], feed)
                print("%d time: %4.4fs, d_loss: %.8f (d_loss_real: %.6f  d_loss_fake: %.6f)" % (steps, time.time() - start, errD, errD1, errD2))

                # if errD2 < 0.5: errA = 1 

                # # 如果D网络的差异太大，需要多学习下G网络
                # for i in range(64):
                #     train_inputs, train_targets = get_next_batch_for_srgan(4)
                #     feed = {inputs: train_inputs, targets: train_targets}

                #     # if errM > 0.5:
                #     # # train G
                #     #     start = time.time() 
                #     #     errM, _ , steps= session.run([g_mse_loss, g_optim_mse, global_step], feed)
                #     #     print("%d time: %4.4fs, g_mse_loss: %.8f " % (steps, time.time() - start, errM))
                #     #     if np.isnan(errM) or np.isinf(errM) :
                #     #         print("Error: cost is nan or inf")
                #     #         return

                #     ## update G
                #     start = time.time()                                
                #     # errG, errM, errV, errA, _, steps = session.run([g_loss, g_mse_loss, g_res_loss, g_gan_loss, g_optim, global_step], feed)
                #     # print("%d time: %4.4fs, g_loss: %.8f (mse: %.6f res: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errV, errA))
                #     errG, errM, errA, _, steps = session.run([g_loss, g_mse_loss, g_gan_loss, g_optim, global_step], feed)
                #     print("%d time: %4.4fs, g_loss: %.8f (mse: %.6f adv: %.6f)" % (steps, time.time() - start, errG, errM, errA))
                #     if np.isnan(errG) or np.isinf(errG) or np.isnan(errA) or np.isinf(errA):
                #         print("Error: cost is nan or inf")
                #         return 

                # # 训练RES
                # for i in range(16):
                #     train_inputs, train_labels, train_seq_len, train_info = get_next_batch_for_res_train(4)
                #     feed = {inputs: train_inputs, labels: train_labels, seq_len: train_seq_len}
                #     start = time.time() 
                #     errR, acc, _ , steps= session.run([res_loss, res_acc, res_optim, global_step], feed)
                #     print("%d time: %4.4fs, res_loss: %.8f, res_acc: %.8f " % (steps, time.time() - start, errR, acc))
                #     if np.isnan(errR) or np.isinf(errR) :
                #         print("Error: cost is nan or inf")
                #         return                       

                # 报告
                if steps > 0 and steps % REPORT_STEPS < 2:
                    train_inputs, train_targets = get_next_batch_for_srgan(4)   
                    p_net_g = session.run(fake_B, {inputs: train_inputs}) 
                    p_net_g = np.squeeze(p_net_g)

                    for i in range(4): 
                        _p_net_g = p_net_g[i]   
                        _train_targets = train_targets[i] 
                        _img = np.vstack((train_inputs[i], _p_net_g, _train_targets)) 
                        cv2.imwrite(os.path.join(curr_dir,"test","F%s_%s.png"%(steps,i)), _img * 255) 

                #     original_list = utils.decode_sparse_tensor(train_labels)
                #     detected_list = utils.decode_sparse_tensor(decoded_list)
                #     if len(original_list) != len(detected_list):
                #         print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
                #             " test and detect length desn't match")
                #     print("T/F: original(length) <-------> detectcted(length)")
                #     acc = 0.
                #     for idx in range(min(len(original_list),len(detected_list))):
                #         number = original_list[idx]
                #         detect_number = detected_list[idx]  
                #         hit = (number == detect_number)          
                #         print("%6s" % hit, list_to_chars(number), "(", len(number), ")")
                #         print("%6s" % "",  list_to_chars(detect_number), "(", len(detect_number), ")")
                #         # 计算莱文斯坦比
                #         import Levenshtein
                #         acc += Levenshtein.ratio(list_to_chars(number),list_to_chars(detect_number))
                #     print("Test Accuracy:", acc / len(original_list))

            # print("Save Model R ...")
            # r_saver.save(session, os.path.join(model_R_dir, "R.ckpt"), global_step=steps)
            print("Save Model D ...")
            d_saver.save(session, os.path.join(model_D_dir, "D.ckpt"), global_step=steps)
            print("Save Model G ...")
            g_saver.save(session, os.path.join(model_G_dir, "G.ckpt"), global_step=steps)
            # try:
            #     ckpt = tf.train.get_checkpoint_state(model_R_dir)
            #     if ckpt and ckpt.model_checkpoint_path:
            #         print("Restore Model R...")
            #         r_saver.restore(session, ckpt.model_checkpoint_path)
            # except:
            #     pass

if __name__ == '__main__':
    train()