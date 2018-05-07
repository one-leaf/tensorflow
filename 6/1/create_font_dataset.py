# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import utils_pil, utils_font, utils_nn
import random
from io import BytesIO

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

if not os.path.exists(os.path.join(curr_dir,"data")):
    os.makedirs(os.path.join(curr_dir,"data"))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def create_font_dataset(filecount=20, filesize=50000):
    for i in range(filecount):        
        TRAINING_TFRECORD_NAME = os.path.join(curr_dir,"data","training_%s.tfrecord"%i)
        if os.path.exists(TRAINING_TFRECORD_NAME): continue
        with tf.python_io.TFRecordWriter(TRAINING_TFRECORD_NAME) as writer:
            font_length = random.randint(5, 200)
            images_count = filesize
            for j in range(images_count):
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

                example = tf.train.Example(features=tf.train.Features(feature={
                    'image': bytes_feature(image.tobytes()),
                    'label': bytes_feature(bytes(text, encoding="utf-8")),
                    'size':  int64_feature(image.size),
                    'font_name': bytes_feature(bytes(font_name, encoding="utf-8")),
                    'font_size': int64_feature(font_size),
                    'font_mode': int64_feature(font_mode),
                    'font_hint': int64_feature(font_hint),
                }))
                writer.write(example.SerializeToString())
                if j%1000==0:
                    print(i, j, 1.0*j/images_count) 
    print('\nFinished writing data to tfrecord files.')


def simple_read():
    TRAINING_TFRECORD_NAME = os.path.join(curr_dir,"data","training.tfrecord")
    for serialized_example in tf.python_io.tf_record_iterator(TRAINING_TFRECORD_NAME):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        size = example.features.feature['size'].int64_list.value
        image = example.features.feature['image'].bytes_list.value[0]
        image = utils_pil.frombytes(tuple(size), image)
        label = str(example.features.feature['label'].bytes_list.value[0],  encoding="utf-8")
        font_size = example.features.feature['font_size'].int64_list.value[0]
        print(label)

if __name__ == '__main__':
    create_font_dataset(1,100)
    # simple_read()
