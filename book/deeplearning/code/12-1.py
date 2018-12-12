# 图片标准化

import tensorflow as tf
import cv2 
import numpy as np
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
image_file = os.path.join(curr_path,'../image/Lenna.jpg')
# 读取图片
img = cv2.imread(image_file)
# 显示原始图片
cv2.imshow("source", img)

# Tensorflow 全局对比度归一化
stand_img = tf.image.per_image_standardization(img)
with tf.Session() as sess:
    result = sess.run(stand_img)
result = np.uint8(result)
cv2.imshow("tensorflow standardization", result)

# Numpy 全局对比度归一化
im = img.astype(np.float32, copy=False)
mean = np.mean(im)
stddev = np.std(im)
adjusted_stddev = max(stddev, 1.0 / np.sqrt(np.array(im.size, dtype=np.float32) ) )
stand_img = (im - mean)/adjusted_stddev
result = np.uint8(stand_img)
cv2.imshow("numpy standardization", result)

# 局部对比对归一化

depth_radius=5
bias=1.0
alpha=0.0001
beta=0.75

# Tensorflow 图像局部归一化
input = np.zeros([1, img.shape[0], img.shape[1], img.shape[2]])
input[0] = img
stand_img = tf.nn.lrn(input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
with tf.Session() as sess:
    result = sess.run(stand_img)
result = np.uint8(result[0])
cv2.imshow("tensorflow local_response_normalization", result)

# Numpy 图像局部归一化
def lrn(input, depth_radius, bias, alpha, beta):
    input_t = input.transpose([2, 0, 1])
    sqr_sum = np.zeros(input_t.shape)
    for i in range(input_t.shape[0]):
        start_idx = i - depth_radius
        if start_idx < 0: start_idx = 0
        end_idx = i + depth_radius + 1
        sqr_sum[i] = sum(input_t[start_idx : end_idx] ** 2)
    return (input_t / (bias + alpha * sqr_sum) ** beta).transpose(1, 2, 0)
im = img.astype(np.float32, copy=False)
stand_img = lrn(im, depth_radius, bias, alpha, beta)
result = np.uint8(stand_img)
cv2.imshow("numpy loc contrast normalization", result)

# Numpy 白化
im = img.astype(np.float32, copy=False)
mean = np.mean(im)
stddev = np.std(im)
# 需要先归一化
input = (im - mean)/stddev
# 转为2维
input = np.transpose(input,[2,0,1])
input = np.reshape(input,[3, 256*256])
# 计算协方差矩阵
sigma = np.dot(input, input.T)/input.shape[1] 
# 奇异分解
U,S,V = np.linalg.svd(sigma) 
# 白化的时候，防止除数为0
epsilon = 0.00001            
# 计算zca白化矩阵    
ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                    
# 白化变换 
result = np.dot(ZCAMatrix, input)   
whitening_img = np.reshape(result,(256,256,1))
result = np.uint8(whitening_img)
cv2.imshow("numpy whitening", result)

cv2.waitKey(0)
cv2.destroyAllWindows()