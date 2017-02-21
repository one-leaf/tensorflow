# coding=utf-8

from PIL import Image
import numpy as np

# 图片转灰度 
def img2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

# 图片转为向量
def img2vec(img,width=-1, height=-1):
    w=img.shape[0]
    h=img.shape[1]
    if width==-1: width=w
    if height==-1: height=h
    vector = np.pad(img,((0,height-h),(0,width-w)), 'constant', constant_values=(255,))  # 在图像上补齐
    vector = vector.flatten() / 255 # 数据扁平化  (vector.flatten()-128)/128  mean为0
    return vector

# 文本转向量
def text2vec(char_set,text):
    text_len = len(text)
    char_set_len = len(char_set)
    vector = np.zeros(text_len*char_set_len)
    for i, c in enumerate(text):
        idx = i * char_set_len + char_set.index(c)
        vector[idx] = 1
    return vector

# 向量转回文本
def vec2text(char_set,vec):
    char_set_len = len(char_set)
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i
        char_idx = c % char_set_len
        text.append(char_set[char_idx])
    return "".join(text)