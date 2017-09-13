# 输出始终比windows下的字体要小，放弃了，改为用 C# 实现做样本

import os
import pygame
from PIL import ImageFont, Image, ImageDraw
from pygame import freetype
import utils
import numpy as np
import cv2

def pygame_font(text,curr_dir):
    pygame.init()
    font = pygame.font.Font(os.path.join(curr_dir, "fonts", "simsun.ttc"), 9)
    rtext = font.render(text, False, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path.join(curr_dir, "data", "pygame.png"))

# 这个最接近windows下的字体渲染
def pygame_freetype_font(text,curr_dir):
    pygame.init()
    freetype.init()
    # font =  freetype.SysFont("simsun",9)
    font = freetype.Font(os.path.join(curr_dir, "fonts", "SIMSUN.TTC"), 9)
    font.antialiased = False
    surf = font.render("中国", fgcolor=(0, 0, 0), bgcolor=(255, 255, 255))[0]
    # im_str = pygame.image.tostring(surf, 'RGB')
    # print(type(imgdata))
    # cv_image = cv2.cv.CreateImageHeader(surf.get_size(), cv.IPL_DEPTH_8U, 3)
    # cv2.cv.SetData(cv_image, imgdata)
    # image = surf.get_view('2').raw
    # image = np.array(image, dtype=np.uint8, copy=True)
    #nparr = np.fromstring(im_str, np.uint8)
    #img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # image = np.fromstring(im_str, np.uint8)
    #utils.show(img_np)
    
    pygame.image.save(surf, os.path.join(curr_dir, "test", "pygame.png"))

def pil_font(text,curr_dir):
    font = ImageFont.truetype(os.path.join(curr_dir, "fonts", "simsun.ttc"),16,index = 0)
    im = Image.new("RGB",(300,200),"White")
    draw = ImageDraw.Draw(im)
    draw.fontmode = "1"
    draw.text( (0,0), '1234567890', font=font, fill="Black")
    draw.text( (0,15), '(abcdefg)', font=font, fill="Black")
    draw.text( (0,30), u'中国', font=font, fill="Black")
    im.save(os.path.join(curr_dir, "test", "pil.png"),dpi=(96,96))

def main():
    curr_dir = os.path.dirname(__file__)
    text = u"1234567890"
    pygame_freetype_font(text,curr_dir)
   # pil_font(text,curr_dir)

if __name__ == '__main__':
    main()