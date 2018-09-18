#coding=utf-8

from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random

import matplotlib.pyplot as plt

# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# 生成字符对应的验证码
def gen_captcha_text_and_image(char_set=number+alphabet+ALPHABET,captcha_size=4,width=160, height=80, saveToFile=None):
    image = ImageCaptcha(width=width,height=height)

    captcha_text = random.sample(list(char_set),captcha_size)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)

    if saveToFile!=None:
        image.write(captcha_text, saveToFile)  # 写到文件

    captcha_image = Image.open(captcha)
    #输出就是图片的矢量
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image

def gen_captcha_image(char_set=number+alphabet+ALPHABET,captcha_size=4,width=160, height=80):
    image = ImageCaptcha(width=width,height=height)
    captcha_text = random.sample(list(char_set),captcha_size)
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text) 
    return Image.open(captcha)

if __name__ == '__main__':
    # 测试
    text, image = gen_captcha_text_and_image(number,4,200,80,"out.png")
    
    print(text)
    print(image.shape)
    plt.imshow(image)
    plt.show()