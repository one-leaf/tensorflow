from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageEnhance 
import random
import matplotlib.pyplot as plt

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
def random_contrast(img, lower = 0.2, upper = 1.8):  
    factor = random.uniform(lower, upper)  
    img = ImageEnhance.Sharpness(img)  
    img = img.enhance(factor)  
    return img  

# 随机亮度  
def random_brightness(img, lower = 0.6, upper = 1.4):  
    factor = random.uniform(lower, upper)  
    img = ImageEnhance.Brightness(img)  
    img = img.enhance(factor)  
    return img  
  
# 随机白平衡 
def random_color(img, lower = 0.6, upper = 1.5):  
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
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)

# 图片缩放
def resize(img, width, height):
    return img.resize((width, height),Image.ANTIALIAS) 

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

def main():
    img = getImage("abced12323","Arial",16)
    show(img)    
    img = random_color(img)
    show(img, None)

if __name__ == '__main__':
    main()