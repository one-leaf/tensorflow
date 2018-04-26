from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageEnhance 
import random
import matplotlib.pyplot as plt
import cv2

# 随机缩小和位置
def random_space(img, minrate=0.5):
    bg = Image.new(img.mode, img.size, 255)
    img = resize(img, random.uniform(minrate, 1))
    w = bg.size[0]-img.size[0]
    h = bg.size[1]-img.size[1]
    x = random.randint(0,w)
    y = random.randint(0,h)
    bg.paste(img, (x,y))
    new_bg = Image.new('L', bg.size, -255)
    new_img = Image.new('L', img.size, 255)
    new_bg.paste(new_img,(x,y))
    return bg, new_bg

# 随机移动位置
def random_space2(img, clear_image, height, minrate=0.5):
    _img = resize_by_height(img, height)
    w,h =_img.size

    bg = Image.new(img.mode, _img.size, 255)
    clear_bg = Image.new(img.mode, _img.size, 255)

    w = bg.size[0]-img.size[0]
    h = bg.size[1]-img.size[1]
    x = 0
    y = 0
    if w>0:
        x = random.randint(0,w)
    if h>0:
        y = random.randint(0,h)

    bg.paste(img, (x,y))
    clear_bg.paste(clear_image, (x,y))
    # new_bg = Image.new('L', bg.size, 255)
    # new_bg.paste(img,(x,y))    
    # return bg, new_bg 
    return bg, clear_bg

# 随机截图
def random_crop(img, width, height):  
    width1 = randint(0, img.size[0] - width )  
    height1 = randint(0, img.size[1] - height)  
    width2 = width1 + width  
    height2 = height1 + height  
    img = img.crop((width1, height1, width2, height2))  
    return img  

# 随机左右翻转  
def random_flip_left_right(img):  
    prob = randint(0,1)  
    if prob == 1:  
        img = img.transpose(Image.FLIP_LEFT_RIGHT)  
    return img  

# 随机对比度变换 
def random_contrast(img, lower = 0.8, upper = 1.2):  
    factor = random.uniform(lower, upper)  
    img = ImageEnhance.Sharpness(img)  
    img = img.enhance(factor)  
    return img  

# 随机亮度  
def random_brightness(img, lower = 0.8, upper = 1.2):  
    factor = random.uniform(lower, upper)  
    img = ImageEnhance.Brightness(img)  
    img = img.enhance(factor)  
    return img  
  
# 随机白平衡 
def random_color(img, lower = 0.8, upper = 1.2):  
    factor = random.uniform(lower, upper)  
    img = ImageEnhance.Color(img)  
    img = img.enhance(factor)  
    return img  

# 显示图片  
def show(img, cmap='gray'):
    plt.imshow(img, cmap = cmap)
    plt.show() 

# 删除边框
def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
    diff = ImageChops.difference(img, bg)
    # diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)

# 获取空白Box
def get_trim_box(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    return bbox
        
# 图片缩放
def resize_by_size(img, size):
    return img.resize(size,Image.ANTIALIAS) 

# 图片缩放
def resize(img, rate):
    width, height = img.size
    width = round(width*rate)
    height = round(height*rate)
    return img.resize((width, height),Image.ANTIALIAS) 

# 图片缩放
def resize_by_height(img, high, antialias=True):
    width, height = img.size
    width = round(width*high/height)
    if antialias:
        return img.resize((width, high),Image.ANTIALIAS)
    else: 
        return img.resize((width, high))
    
# RGBA to RGB
def convert_to_rgb(img):
    mode = img.mode
    if mode not in ('L', 'RGB'):
        if mode == 'RGBA':
            # 透明图片需要加白色底
            alpha = img.split()[3]
            bgmask = alpha.point(lambda x: 255-x)
            img = img.convert('RGB')
            img.paste((255,255,255), None, bgmask)
        else:
            img = img.convert('RGB')
    return img

# RGB to 灰度
def convert_to_gray(img):
    return img.convert('L')

# 灰度图片转黑白，这里都是干净图片，所以不能 cv2.THRESH_OTSU 动态转，需要是固定值转
def convert_to_bw(img_gray_array):
    thresh, img_bw = cv2.threshold(img_gray_array, 50, 255, cv2.THRESH_BINARY)
    return img_bw

def frombytes(img_bytes):
    return Image.frombytes('RGBA', img_bytes)

def main():
    img = getImage("abced12323","Arial",16)
    show(img)    
    img = random_color(img)
    show(img, None)

if __name__ == '__main__':
    main()