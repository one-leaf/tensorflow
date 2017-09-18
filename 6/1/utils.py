import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# 显示图片,esc 关闭，参数是 np.array 类型
def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

# img 参数是 np.array 类型 输入是二值化并且反色的图片
# 用于清除表格线
def clearImg(adaptive_binary_inv):
    # 清除竖线
    _sum=np.sum(adaptive_binary_inv,axis=0)
    _mean = 255 * adaptive_binary_inv.shape[0]
    for i,x in enumerate(_sum):
        if x>_mean*0.8:
            adaptive_binary_inv[:,i]=0
    # 清除横线
    _sum=np.sum(adaptive_binary_inv,axis=1)
    _mean = 255 * adaptive_binary_inv.shape[1]
    for i,x in enumerate(_sum):
        if x>_mean*0.8:
            adaptive_binary_inv[i,:]=0

# 图片转灰度, 参数是 np.array 类型
def img2gray(img_color):
    return cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 将图片转为黑白2色，参数是 np.array 类型
# 为了方便计算，需要反色
# 后面的方法更好一些，会保留一些轮廓信息
def img2bw(img_gray):
    thresh, img_bw = cv2.threshold(img_gray, 192, 255, cv2.THRESH_BINARY_INV)
    # thresh, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 11)
    # 去噪点，实际测试不需要
    # kernel = np.ones((3, 3), np.uint8)
    # open = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel, iterations=2)
    # show(img_bw)
    return img_bw

# 图片转为向量, img 参数是 np.array 类型
def img2vec(img, height=-1, width=-1, value=0):
    h=img.shape[0]
    w=img.shape[1]
    if width==-1: width=w
    if height==-1: height=h
    if h>height or w>width:
        raise "image size too large"
    vector = np.pad(img,((0,height-h),(0,width-w)), 'constant', constant_values=(value,))  # 在图像上补齐
    vector = vector.flatten() / 255 # 数据扁平化  (vector.flatten()-128)/128  mean为0
    return vector

# 图片分割，按水平投影分割
# img_gray 传入的灰度图像
def splitImg(img_gray):
    # 将灰度图二值化，并反色
    adaptive_binary_inv=img2bw(img_gray)
    # thresh, adaptive_binary_inv = cv2.threshold(img_gray, 192, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 清除多余的线段
    clearImg(adaptive_binary_inv)

    h_sum = np.sum(adaptive_binary_inv, axis=1)
    peek_ranges = extract_peek_ranges_from_array(h_sum,3,5)

    images=[]
    for i, peek_range in enumerate(peek_ranges):
        x = 0
        y = peek_range[0]
        w = adaptive_binary_inv.shape[1]
        h = peek_range[1] - y
        # 删除前面和后面的空白区域
        w_sum = np.sum(adaptive_binary_inv[y: y + h, x: x + w], axis=0)
        for s in w_sum:
            if s==0:
                x += 1
            else:
                break
        w = adaptive_binary_inv.shape[1] - x 
        for s in w_sum[::-1]:
            if s==0:
                w -= 1
            else:
                break
        images.append(img_gray[y: y + h , x: x + w ])
    return images
    
# 从一个数组抓到分割点
# minimun_val 最小分割的最小值
# minimun_range 最小分割的长度
def extract_peek_ranges_from_array(array_vals, minimun_val=0, minimun_range=5):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def readImgFile(filename):
    _img = cv2.imread(filename, 0)
    _img = img2bw(_img)
    return _img

# img_gray 传入的灰度图像
# minArea 最小的区域面积
# x,y,w,h 过滤器
# 返回 (img,rect)
def getGrids(img_gray,minArea=0,x=0,y=0,w=0,h=0):
    # 将灰度图像二值化
    #adaptive_binary= cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    thresh, adaptive_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, adaptive_binary = cv2.threshold(adaptive_binary,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(adaptive_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    images=[]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        _x,_y,_w,_h = cv2.boundingRect(cnt)
        if minArea>0 and area < minArea: continue
        if x>0 and x!=_x: continue
        if y>0 and y!=_y: continue
        if w>0 and w!=_w: continue
        if h>0 and h!=_h: continue
        # cv2.drawContours(adaptive_binary, cnt, -1, (0,255,0), 3)
        # show(adaptive_binary)
        # 高度和宽度减了一个像素，防止灰度边框
        img = img_gray[_y:_y+_h-1,_x:_x+_w-1]
        images.append((img,(_x,_y,_w,_h)))
    return images

# 装载图片，并分解为待识别图像
def loadImage(filename):
    img = cv2.imread(filename, 0)
    if img.shape != (1123,794):
        raise "不是进口商检单"

    result_images =[]

    images = getGrids(img,1000,h=210)
    # 按 rect 的 x 排序
    sorted_images = sorted(images, key = lambda image: image[1][0])
    for img,rect in sorted_images:
        split_images = splitImg(img)
        b_w_split_images = []
        for split_image in split_images:
            b_w_split_images.append(img2bw(split_image))
        result_images.append(b_w_split_images)        

#            show(split_image)
#            print(split_image.shape)
    return result_images   


def main():
    curr_dir = os.path.dirname(__file__)
    need_ocr_images = loadImage(os.path.join(curr_dir,'test','0.jpg'))

if __name__ == '__main__':
    main()