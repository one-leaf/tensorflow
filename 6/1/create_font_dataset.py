# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import utils_pil, utils_font, utils_nn
import random

curr_dir = os.path.dirname(__file__)

# 所有 unicode CJK统一汉字（4E00-9FBB） + ascii的字符加 + ctc blank
# https://zh.wikipedia.org/wiki/Unicode
# https://zh.wikipedia.org/wiki/ASCII
ASCII_CHARS = [chr(c) for c in range(32,126+1)]
#ZH_CHARS = [chr(c) for c in range(int('4E00',16),int('9FBB',16)+1)]
#ZH_CHARS_PUN = ['。','？','！','，','、','；','：','「','」','『','』','‘','’','“','”',\
#                '（','）','〔','〕','【','】','—','…','–','．','《','》','〈','〉']

CHARS = ASCII_CHARS #+ ZH_CHARS + ZH_CHARS_PUN
# CHARS = ASCII_CHARS
CLASSES_NUMBER = len(CHARS) + 1 

ENGFontNames, CHIFontNames = utils_font.get_font_names_from_url()
print("EngFontNames", ENGFontNames)
print("CHIFontNames", CHIFontNames)
AllFontNames = ENGFontNames + CHIFontNames
AllFontNames.remove("方正兰亭超细黑简体")
AllFontNames.remove("幼圆")
AllFontNames.remove("方正舒体")
AllFontNames.remove("方正姚体")
AllFontNames.remove("华文新魏")
AllFontNames.remove("Impact")
AllFontNames.remove("Gabriola")

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 

TRAINING_TFRECORD_NAME = os.path.join(curr_dir,"data","training.tfrecord")
if not os.path.exists(os.path.join(curr_dir,"data")):
    os.makedirs(os.path.join(curr_dir,"data"))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def create_font_dataset():
    with tf.python_io.TFRecordWriter(TRAINING_TFRECORD_NAME) as writer:
        font_length = random.randint(5, 200)
        images_count = 1000000
        for i in range(images_count):
            font_name = random.choice(AllFontNames)
            if random.random()>0.5:
                font_size = random.randint(9, 49)    
            else:
                font_size = random.randint(9, 15) 
            font_mode = random.choice([0,1,2,4]) 
            # hint 2 在小字体下会断开笔画，人眼都无法识别
            if font_size>=14:
                font_hint = random.choice([0,1,2,3,4,5])  
            else:
                font_hint = random.choice([0,1,3,4,5]) 

            text  = utils_font.get_words_text(CHARS, eng_world_list, font_length)
            text = text + " " + "".join(random.sample(CHARS, random.randint(1,5)))
            text = text.strip()

            image = utils_font.get_font_image_from_url(text, font_name, font_size, font_mode, font_hint)
            image = image.tobytes(encoder_name="png")

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': bytes_feature(image),
                'image/labels': bytes_feature(bytes(text, encoding="utf-8")),
                'image/font_name': bytes_feature(bytes(font_name, encoding="utf-8")),
                'image/font_size': int64_feature(font_size),
                'image/font_mode': int64_feature(font_mode),
                'image/font_hint': int64_feature(font_hint),
            }))
            writer.write(example.SerializeToString())
            if i%1000==0:
                print(i, 1.0*i/images_count) 
    print('\nFinished writing data to tfrecord files.')

if __name__ == '__main__':
    create_font_dataset()
