import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 显示图片,esc 关闭，参数是 np.array 类型
def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

# img 参数是 np.array 类型 输入是灰度后的数据 2D
# 用于清除表格线
def clearImg(img):
    # 清除竖线
    _sum=np.sum(img,axis=0)
    _mean = 255 * img.shape[0]
    for i,x in enumerate(_sum):
        if x>_mean*0.8:
            img[:,i]=0

    # 清除横线
    _sum=np.sum(img,axis=1)
    _mean = 255 * img.shape[1]
    for i,x in enumerate(_sum):
        if x>_mean*0.8:
            img[i,:]=0

# 图片转灰度, 参数是 np.array 类型
def img2gray(img_color):
    return cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 将图片转为黑白2色，参数是 np.array 类型
# 为了方便计算，需要反色
# 后面的方法更好一些，会保留一些轮廓信息
def img2bw(img_gray):
    # thresh, img_bw = cv2.threshold(img_gray, 192, 255, cv2.THRESH_BINARY_INV)
    # thresh, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 11)
    # 去噪点，实际测试不需要
    # kernel = np.ones((3, 3), np.uint8)
    # open = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel, iterations=2)
    # show(img_bw)
    return img_bw

# 图片转为向量, img 参数是 np.array 类型
def img2vec(img, width=-1, height=-1):
    w=img.shape[0]
    h=img.shape[1]
    if width==-1: width=w
    if height==-1: height=h
    vector = np.pad(img,((0,height-h),(0,width-w)), 'constant', constant_values=(255,))  # 在图像上补齐
    vector = vector.flatten() / 255 # 数据扁平化  (vector.flatten()-128)/128  mean为0
    return vector

# 图片分割，先按水平投影分割，然后按垂直投影分割
def splitImg(img):
    h_sum = np.sum(img, axis=1)
    # plt.plot(h_sum, range(h_sum.shape[0]))
    # plt.gca().invert_yaxis()
    # plt.show()
    peek_ranges = extract_peek_ranges_from_array(h_sum,10,5)
    # line_seg_adaptive_threshold = np.copy(img)
    # for i, peek_range in enumerate(peek_ranges):
    #     x = 0
    #     y = peek_range[0]
    #     w = line_seg_adaptive_threshold.shape[1]
    #     h = peek_range[1] - y
    #     pt1 = (x, y)
    #     pt2 = (x + w, y + h)
    #     cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
    # cv2.imshow('line image', line_seg_adaptive_threshold)
    # cv2.waitKey(0)

    v_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = img[start_y:end_y, :]
        v_sum = np.sum(line_img, axis=0)
        # plt.plot(v_sum, range(v_sum.shape[0]))
        # plt.gca().invert_yaxis()
        # plt.show()
        print(v_sum,)
        v_peek_ranges = extract_peek_ranges_from_array(v_sum,max(v_sum)*0.1,1)
        v_peek_ranges2d.append(v_peek_ranges)

    print(v_peek_ranges2d)

    color = (255, 0, 0)
    line_seg_adaptive_threshold = np.copy(img)
    for i, peek_range in enumerate(peek_ranges):
        for v_range in v_peek_ranges2d[i]:
            x = v_range[0]
            y = peek_range[0]
            w = v_range[1] - x
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(img, pt1, pt2, color)
    cv2.imshow('char image', img)
    cv2.waitKey(0)

    # print(ranges)

# 从一个数组抓到分割点
# minimun_val 最小分割的最小值
# minimun_range 最小分割的长度
def extract_peek_ranges_from_array(array_vals, minimun_val=5, minimun_range=5):
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

def main():
    curr_dir = os.path.dirname(__file__)
    # image = Image.open(os.path.join(curr_dir,"data/13.png"))

    # _img = image2array(image)
    # _img = img2gray(_img)

    # 直接用 cv2 按灰度读取
    _img = cv2.imread(os.path.join(curr_dir,"test/N-05.png"), 0)
    _img = img2bw(_img)
    clearImg(_img)
    splitImg(_img)

if __name__ == '__main__':
    main()