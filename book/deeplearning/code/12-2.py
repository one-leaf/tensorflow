# 图片标准化

import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 
import numpy as np
import os
import math
import random

curr_path = os.path.dirname(os.path.realpath(__file__))
image_file = os.path.join(curr_path,'../image/Lenna.jpg')
# 读取图片，cv2 的读取颜色通道是 BGR
img = cv2.imread(image_file)
# 将图片转为 RGB 
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

with tf.Session() as sess:
    plt.figure()
    plt.subplot(231)
    plt.imshow(img)
    plt.title("source")

    plt.subplot(232)
    tf_img = tf.random_crop(img,[200,200,3])
    plt.imshow(sess.run(tf_img))
    plt.title("random_crop")

    plt.subplot(233)
    tf_img = tf.image.random_flip_left_right(img)
    plt.imshow(sess.run(tf_img))
    plt.title("random_flip_left_right")

    plt.subplot(234)
    tf_img = tf.image.random_flip_up_down(img)
    plt.imshow(sess.run(tf_img))
    plt.title("random_flip_up_down")

    plt.subplot(235)
    # 角度转弧度
    degress = math.radians(random.randint(0,360))
    tf_img = tf.contrib.image.rotate(img, degress)
    plt.imshow(sess.run(tf_img))
    plt.title("random_flip_up_down")

    # 非线性变形
    plt.subplot(236)
    input = tf.convert_to_tensor_or_sparse_tensor(img, dtype=tf.float32)
    input = input/255.
    input = tf.reshape(input, [1, img.shape[0], img.shape[1], 3])

    filter = np.random.standard_normal([5, 5, 3, 2])
    flow = tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')

    tf_img = tf.contrib.image.dense_image_warp(input, flow)
    tf_img = sess.run(tf_img)[0]
    plt.imshow(tf_img)
    plt.title("dense_image_warp")

    ###############################

    plt.figure()
    plt.subplot(231)
    plt.imshow(img)
    plt.title("source")

    plt.subplot(232)
    tf_img = tf.image.random_brightness(img,max_delta=0.8)
    plt.imshow(sess.run(tf_img))
    plt.title("random_brightness")

    plt.subplot(233)
    tf_img = tf.image.random_contrast(img,lower=0.2,upper=1.8)
    plt.imshow(sess.run(tf_img))
    plt.title("random_contrast")

    plt.subplot(234)
    tf_img = tf.image.random_hue(img,max_delta=0.3)
    plt.imshow(sess.run(tf_img))
    plt.title("random_hue")

    plt.subplot(235)
    tf_img = tf.image.random_saturation(img,lower=0.2,upper=1.8)
    plt.imshow(sess.run(tf_img))
    plt.title("random_saturation")

    plt.subplot(236)
    tf_img = tf.image.random_jpeg_quality(img, min_jpeg_quality=50, max_jpeg_quality=98)
    plt.imshow(sess.run(tf_img))
    plt.title("random_jpeg_quality")

plt.show()

