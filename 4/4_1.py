# coding=utf-8
'''
这个是从训练模型中加载数据
'''
from generate_captcha import gen_captcha_text_and_image as captcha
from utils import img2gray, img2vec, text2vec, vec2text
import numpy as np
import tensorflow as tf
import os

out_dir = os.path.dirname(__file__)
log_dir = os.path.join(out_dir, "logs")
if not os.path.exists(log_dir):
    print("error: can't dind log dir")
    exit()

checkpoint_prefix = os.path.join(log_dir, "model.ckpt")

# 找到最新的运算模型文件
metaFile= sorted(
    [
        (x, os.path.getctime(os.path.join(log_dir,x)))                  
        for x in os.listdir(log_dir) if x.endswith('.meta')  
    ],
    key=lambda i: i[1])[-1][0]

sess = tf.Session()

saver = tf.train.import_meta_graph(os.path.join(log_dir,metaFile))
ckpt = tf.train.get_checkpoint_state(log_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("error: can't load checkpoint data")
    exit()

text, image = captcha(char_set="0123456789", captcha_size=4, width=200, height=80)

# 如果是彩色照片，直接转灰度
#if image.shape[2] != 1:
#    image = tf.image.rgb_to_grayscale(image)

# 如果图片不符合训练的大小，需要调整图片大小    
image = img2vec(img2gray(image)) #tf.image.resize_images(image, (200, 80))
#imagedata = tf.reshape(image, (1, 200, 80, 1))
x = tf.get_default_graph().get_tensor_by_name('x:0')
prediction = tf.get_default_graph().get_tensor_by_name('stack:0')
#imagedata = image.eval(session=sess)
#print(imagedata.shape)
sess.run([prediction], feed_dict={x: image})