import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw, ImageFont, ImageChops

# 删除边框
def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)

# 按高度缩放图片,img_shape=(height,width)
def resize(img,height=28):
    width = round(height*img.shape[1]/img.shape[0])
    # print(img.shape[0],img.shape[1],width,height)
    return cv2.resize(img,(width,height),interpolation=cv2.INTER_NEAREST)

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

# 转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
 
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result
    
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = spars_tensor[1][m]
        decoded.append(str)
    return decoded


# 显示图片,esc 关闭，参数是 np.array 类型
def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def pltshow(img, cmap = 'gray'):
    plt.imshow(img, cmap)
    plt.show()

# 保存图片
def save(img,filename):
    cv2.imwrite(filename,img)

# 清除背景色
def clearBackgroundColor(gray_image, replace_Color=255):
    f_img = gray_image.flatten().astype(int)
    counts = np.bincount(f_img)
    v = np.argmax(counts)
    same = 0
    if v== replace_Color:
        return gray_image
    if len(gray_image.shape) == 3:
        if v == gray_image[0,0,0]: same+=1
        if v == gray_image[0,-1,0]: same+=1
        if v == gray_image[-1,0,0]: same+=1
        if v == gray_image[-1,-1,0]: same+=1
    if len(gray_image.shape) == 2:
        if v == gray_image[0,0]: same+=1
        if v == gray_image[0,-1]: same+=1
        if v == gray_image[-1,0]: same+=1
        if v == gray_image[-1,-1]: same+=1        
    if same>=3:
        img = gray_image - v
        zero_mask = img == 0
        gray_image[zero_mask] = replace_Color
    return gray_image

# img 参数是 np.array 类型 输入是正常灰度图片
# 用于清除表格线
def clearImgGray(gray_image):
    adaptive_binary_inv=img2bwinv(gray_image)
    # 清除竖线
    _sum=np.sum(adaptive_binary_inv,axis=0)
    _mean = 255 * adaptive_binary_inv.shape[0]
    for i,x in enumerate(_sum):
        if x>_mean*0.95:
            gray_image[:,i]=255
    # 清除横线
    _sum=np.sum(adaptive_binary_inv,axis=1)
    _mean = 255 * adaptive_binary_inv.shape[1]
    for i,x in enumerate(_sum):
        if x>_mean*0.8:
            gray_image[i,:]=255

    return dropZeroEdgesGray(gray_image)


# img 参数是 np.array 类型 输入是二值化并且反色的图片
# 用于清除表格线
def clearImg(adaptive_binary_inv):
    result = np.copy(adaptive_binary_inv)
    # 清除竖线
    _sum=np.sum(adaptive_binary_inv,axis=0)
    _mean = 255 * adaptive_binary_inv.shape[0]
    for i,x in enumerate(_sum):
        if x>_mean*0.95:
            result[:,i]=0
    # 清除横线
    _sum=np.sum(adaptive_binary_inv,axis=1)
    _mean = 255 * adaptive_binary_inv.shape[1]
    for i,x in enumerate(_sum):
        if x>_mean*0.8:
            result[i,:]=0
    return result

# 图片转灰度, 参数是 np.array 类型
def img2gray(img_color):
    return cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 将图片转为黑白2色，参数是 np.array 类型
# 为了方便计算，需要反色
# 后面的方法更好一些，会保留一些轮廓信息
def img2bwinv(img_gray):
    thresh, img_bw = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresh, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    # img_bw = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # 去噪点，实际测试不需要
    # kernel = np.ones((3, 3), np.uint8)
    # open = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel, iterations=2)
    # show(img_bw)
    return img_bw

# 将原图平铺到目标图
# ###########  ==>  ######
#                   ######
#                   ###
#
def square_img(srcimg, dstimg, height=None):
    s_h,s_w=srcimg.shape
    if height == None: height = s_h
    d_h,d_w=dstimg.shape
    for w in range(s_w):
        for h in range(s_h):
            l = w//d_w
            dstimg[l*height+h][w%d_w] = srcimg[h][w]  
    return dstimg           

# 将原图中的数据取出来并拼接为完整的
#  ######  ==>  ##############
#  ######
#  ###
#
def unsquare_img(srcimg, height):
    s_h,s_w=srcimg.shape
    d_w = s_w * (s_h//height)
    dstimg = np.zeros((height,d_w))
    for w in range(d_w):
        for h in range(height):
            l = w//s_w
            dstimg[h][w]=srcimg[l*height+h][w%s_w]
    return dstimg

# 图片转为向量, img 参数是 np.array 类型
def img2vec(img, height=-1, width=-1, value=0, flatten=True):
    h=img.shape[0]
    w=img.shape[1]
    if width==-1: width=w
    if height==-1: height=h
    if h>height or w>width:
        raise Exception("image size too large, src size: %s,%s dst size: %s,%s" % (w,h,width,height))
    vector = np.pad(img,((0,height-h),(0,width-w)), 'constant', constant_values=(value,))  # 在图像上补齐
    if flatten:
        vector = vector.flatten()
    # vector = vector / 255 # 数据扁平化  (vector.flatten()-128)/128  mean为0
    return vector

# 清除边缘 输入为灰度图片
def dropZeroEdgesGray(img_gray):
    img=img2bwinv(img_gray)
    return dropZeroEdges(img, img_gray)

# 清除边缘 输入为反色图片
def dropZeroEdges(img_inv, img_gray=[], min_rate=0):
    temp = np.copy(img_inv)

    if min_rate>0:
        h_sums = np.sum(temp, axis=1)
        avg = np.average(np.trim_zeros(h_sums))
        h_sums_avg = h_sums*1.0/avg
        for i in range(len(h_sums)):
            if h_sums_avg[i] < min_rate:
                temp[i] = 0
            else:
                break
        for i in reversed(range(len(h_sums))):
            if h_sums_avg[i] < min_rate:
                temp[i] = 0
            else:
                break

        w_sums = np.sum(temp, axis=0)
        avg = np.average(np.trim_zeros(w_sums))
        w_sums_avg = w_sums*1.0/avg
        min_rate = min_rate * 2
        for i in range(len(w_sums)):
            if w_sums_avg[i] < min_rate or (w_sums_avg[i+1] < min_rate and w_sums_avg[i+2] < min_rate and w_sums_avg[i+3] < min_rate):
                temp[:,i] = 0
            else:
                break
        for i in reversed(range(len(w_sums))):
            if w_sums_avg[i] < min_rate or (w_sums_avg[i-1] < min_rate and w_sums_avg[i-2] < min_rate and w_sums_avg[i-3] < min_rate):
                temp[:,i] = 0
            else:
                break

    true_points = np.argwhere(temp)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    if top_left[0] == bottom_right[0] or top_left[1] == bottom_right[1] : return img_inv
    # print(img_inv.shape)
    # print(top_left[0], bottom_right[0], top_left[1], bottom_right[1])
    if len(img_gray)>0 :
        return img_gray[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    else:
        return img_inv[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

# 图片分割，按水平投影分割
# img_gray 传入的灰度图像
def splitImg(img_gray):
    # 将灰度图二值化，并反色
    adaptive_binary_inv=img2bwinv(img_gray)
    # pltshow(adaptive_binary_inv)
    # thresh, adaptive_binary_inv = cv2.threshold(img_gray, 192, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 清除多余的线段
    oldimg = adaptive_binary_inv
    adaptive_binary_inv = clearImg(adaptive_binary_inv)
    # pltshow(np.vstack([oldimg,adaptive_binary_inv]))

    h_sum = np.sum(adaptive_binary_inv, axis=1)
    peek_ranges = extract_peek_ranges_from_array(h_sum,0,5)
    images=[]
    for i, peek_range in enumerate(peek_ranges):
        x = 0
        y = peek_range[0]
        w = adaptive_binary_inv.shape[1]
        h = peek_range[1] - y + 1
        # 删除前面和后面的空白区域
       
        w_sum = np.sum(adaptive_binary_inv[y - 1: y + h + 1, x: x + w + 1], axis=0)
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
        images.append(img_gray[y: y + h , x: x + w + 1 ])
    return images
    
# 从一个数组抓到分割点
# minimun_val 最小分割的最小值
# minimun_range 最小分割的长度
# end_i包含最后一位
def extract_peek_ranges_from_array(array_vals, minimun_val=0, minimun_range=5):
    start_i = None
    end_i = None
    peek_ranges = []
    zero_count = 0
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            end_i = i
        elif val <= minimun_val and start_i is not None:
            if end_i - start_i >= minimun_range and zero_count >= 1:
                peek_ranges.append((start_i, end_i))
                start_i = None
                end_i = None
                zero_count = 0
            else:
                zero_count += 1
        elif val <= minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    if start_i is not None and end_i is not None:
        peek_ranges.append((start_i, end_i))
    return peek_ranges

def readImgFile(filename):
    _img = cv2.imread(filename, 0)
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

def getMaxContours(img):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h =(0,0,0,0)
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            x,y,w,h = cv2.boundingRect(cnt)
    return x,y,w,h


def renderFontBypyGame(font_file, font_size, text):
    from pygame import freetype
    import pygame
    pygame.init()
    freetype.init()
    try:
        font = freetype.Font(font_file, font_size)
        font.antialiased = random.random()>0.5 
        styles = [freetype.STYLE_DEFAULT,freetype.STYLE_NORMAL,freetype.STYLE_OBLIQUE,
                    freetype.STYLE_STRONG,freetype.STYLE_UNDERLINE,freetype.STYLE_WIDE]
        while True:
            try:
                style =random.choice(styles)
                rtext = font.render(text, (0, 0, 0), (255, 255, 255), style=style)[0]
                break
            except:
                print("rechoise font style for",font_file)           
        data = pygame.image.tostring(rtext, 'RGBA')
        _img = Image.frombytes("RGBA",rtext.get_size(),data)
        size = _img.size
        img=Image.new("RGBA",(size[0]+100,size[1]+100),(255,255,255))
        img.paste(_img,(50,50))
        return img
    except:
        raise Exception("Error font %s" % font_file)        

def renderFontByPIL(font_file, font_size, text):
    try:
        font = ImageFont.truetype(font_file, font_size, index = 0)
        size = font.getsize(text)
        img=Image.new("RGBA",(size[0]+100,size[1]+100),(255,255,255))
        draw = ImageDraw.Draw(img)
        fontmode = random.choice(["1", "P", "I", "F", "L"])
        draw.fontmode=fontmode
        draw.text((50,50),text,fill='black',font=font)
        return img
    except:
        raise Exception("Error font %s" % font_file)    

def renderNormalFontByPIL(font_file, font_size, text):
    try:
        font = ImageFont.truetype(font_file, font_size, index = 0)
        size = font.getsize(text)
        img=Image.new("RGBA",(size[0]+100,size[1]+100),(255,255,255))
        draw = ImageDraw.Draw(img)
        draw.text((50,50),text,fill='black',font=font)
        return img
    except:
        raise Exception("Error font %s" % font_file) 

def getImage(CHARS, font_file, image_height=16, font_length=30, font_size=12, word_dict=None, is_Debug=False):
    if is_Debug:
        print(font_file,font_size)
    text=''
    n = random.random()
    if n<0.1:
        for i in range(font_length):
            text += random.choice("123456789012345678901234567890-./$,:()+-*=><")
    elif n<0.5 and n>=0.1:
        for i in range(font_length):
            text += random.choice(CHARS)        
    else:
        while len(text)<font_length:
            word = random.choice(word_dict)
            _word=""
            for c in word:
                if c in CHARS:
                    _word += c
            text = text+" "+_word.strip()
    text=text.strip()

    r = random.random()
    if r>=0.5:
       img = renderFontBypyGame(font_file, font_size, text)        
    else:
       img = renderFontByPIL(font_file, font_size, text)

    if is_Debug:
        return text, img


    # 缩放一下
    img = trim(img)
    w,h=img.size
    # print(img.size)
    _h = random.randint(9,64)
    _w = round(w * _h / h)
    img = img.resize((_w,_h), Image.ANTIALIAS)


    # 做轻微的旋转 +- 1.5度
    # w,h=img.size
    # rot = img.rotate((random.random()-0.5)*3, expand=1)
    # img=Image.new("RGBA",(w+50,h+50),(255,255,255))
    # img.paste(rot,rot)

    # 轻微扭曲
    # params = [
    #     1 - float(random.randint(1,2)) / 1000,
    #     0,
    #     0,
    #     0,
    #     1 - float(random.randint(1,10)) /1000,
    #     float(random.randint(1,2)) / 5000,
    #     0.0002,
    #     float(random.randint(1,2)) / 5000
    # ]
    # img = img.transform((size[0]+100,size[1]+100), Image.PERSPECTIVE, params)

   # img = utils.resize(utils.dropZeroEdges(utils.img2bwinv(utils.img2gray(np.asarray(img)))), 32) 
    img = np.asarray(img)
    img = clearBackgroundColor(img)
   
    img = 1 - img2gray(img)/255.   
    #img = img2bwinv(img)
    img = dropZeroEdges(img)

    # 添加噪点
    # filter = np.random.random(img.shape) - 0.8
    # filter = np.maximum(filter, 0) 
    # img = img + filter * 2
    # imin, imax = img.min(), img.max()
    # img = (img - imin)/(imax - imin)

    img = resize(img, image_height)
    return text, img

# def main():
#     curr_dir = os.path.dirname(__file__)
#     FontDir = os.path.join(curr_dir,"fonts")
#     FontNames = []    
#     # fontName = os.path.join(curr_dir,"fonts","simsun.ttc")
#     for name in os.listdir(FontDir):
#         fontName = os.path.join(FontDir, name)
#         if fontName.lower().endswith('ttf') or \
#            fontName.lower().endswith('ttc') or \
#            fontName.lower().endswith('otf'):
#            FontNames.append(fontName)

#     fontName = random.choice(FontNames)
#     eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 
#     ASCII_CHARS = [chr(c) for c in range(32,126+1)]
#     lable,img = getImage(ASCII_CHARS,fontName,image_height=32, font_length=50, \
#             font_size=64,word_dict=eng_world_list,is_Debug=False)
#     print(lable)
#     #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

#     plt.imshow(img, cmap = 'gray',)
#     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#     plt.show()
    
#     # cv2.imshow(lable,np.asarray(img))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()    

def main():
    img = Image.open("D://S4_0.png")
    img = np.array(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    show(img)


if __name__ == '__main__':
    main()