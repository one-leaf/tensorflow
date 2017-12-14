import urllib,json,io
import urllib.parse , urllib.request
import random
import utils_pil
from PIL import Image
import json

def http(url,param=None):
    if param !=None:
        paramurl = urllib.parse.urlencode(param)
        url = "%s?%s"%(url,paramurl)
        r = urllib.request.urlopen(url, timeout=30)
    else:    
        r = urllib.request.urlopen(url, timeout=30)
    return r.read()

def get_font_names_from_url():
    r = http('http://192.168.2.113:8888/')
    fonts = json.loads(r.decode('utf-8'))
    ENGFontNames = fonts['eng']
    CHIFontNames = fonts['chi']
    return ENGFontNames, CHIFontNames

# 从字体服务器上获取字体图片
def get_font_image_from_url(text, font_name, font_size, fontmode=None, fonthint=None):
    params= {}
    params['text'] = text
    params['fontname'] = font_name
    params['fontsize'] = font_size
    # params['fontmode'] = random.choice([0,1,2,4,8])
    if fontmode == None:
        params['fontmode'] = random.choice([0,1,2,4])
    else:
        params['fontmode'] = fontmode
    if fonthint == None:
        params['fonthint'] = random.choice([0,1,2,3,4,5])
    else:
        params['fonthint'] = fonthint  
    r = http('http://192.168.2.113:8888/',params)
    img = Image.open(io.BytesIO(r))
    img = utils_pil.convert_to_rgb(img)
    img = utils_pil.trim(img)
    return img

# 获得随机字符串
def get_random_text(CHARS, word_dict, font_length):
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
    text = text.strip()
    return text

def get_font_image_from_pygame(font_file, font_size, text):
    import pygame
    from pygame import freetype
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

def get_font_image_from_pil(font_file, font_size, text, random_font_mode=False):
    try:
        font = ImageFont.truetype(font_file, font_size, index = 0)
        size = font.getsize(text)
        img  = Image.new("RGBA",(size[0]+100,size[1]+100),(255,255,255))
        draw = ImageDraw.Draw(img)
        if random_font_mode:
            fontmode = random.choice(["1", "P", "I", "F", "L"])
            draw.fontmode=fontmode
        draw.text((50,50),text,fill='black',font=font)
        return img
    except:
        raise Exception("Error font %s" % font_file)    

def get_font_image_and_text_from_local(CHARS, font_file, font_length=30, font_size=12, word_dict=None):
    text = get_random_text(CHARS, word_dict, font_length)    
    r = random.random()
    if r>=0.5:
       img = renderFontBypyGame(font_file, font_size, text)        
    else:
       img = renderFontByPIL(font_file, font_size, text, random_font_mode=True)
    img = utils_pil.convert_to_rgb(img)
    img = utils_pil.trim(img) 
    return text, img

def get_font_image_and_text_from_url(CHARS, font_name, font_length=30, font_size=12, word_dict=None):
    text = get_random_text(CHARS, word_dict, font_length)
    img  = get_font_image_from_url(text, font_name, font_size)   
    return text, img

def add_noise(img):
    img = utils_pil.random_brightness(img)
    img = utils_pil.random_color(img)
    img = utils_pil.random_contrast(img)
    return img