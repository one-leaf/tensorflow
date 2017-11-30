# coding=utf-8
# 中文OCR学习，尝试多层

import tensorflow as tf
import numpy as np
import os
import utils
import time
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import tensorflow.contrib.slim as slim
import math
import urllib,json,io

curr_dir = os.path.dirname(__file__)

image_height = 32

# LSTM
# num_hidden = 4
# num_layers = 1

# 所有 unicode CJK统一汉字（4E00-9FBB） + ascii的字符加 + ctc blank
# https://zh.wikipedia.org/wiki/Unicode
# https://zh.wikipedia.org/wiki/ASCII
ASCII_CHARS = [chr(c) for c in range(32,126+1)]
#ZH_CHARS = [chr(c) for c in range(int('4E00',16),int('9FBB',16)+1)]
#ZH_CHARS_PUN = ['。','？','！','，','、','；','：','「','」','『','』','‘','’','“','”',\
#                '（','）','〔','〕','【','】','—','…','–','．','《','》','〈','〉']

CHARS = ASCII_CHARS #+ ZH_CHARS + ZH_CHARS_PUN
# CHARS = ASCII_CHARS

#初始化学习速率
LEARNING_RATE_INITIAL = 1e-4
# LEARNING_RATE_DECAY_FACTOR = 0.9
# LEARNING_RATE_DECAY_STEPS = 2000
REPORT_STEPS = 500
MOMENTUM = 0.9

BATCHES = 64
BATCH_SIZE = 32
TRAIN_SIZE = BATCHES * BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE
POOL_COUNT = 3
POOL_SIZE  = round(math.pow(2,POOL_COUNT))

# 增加 Highway 网络
def addHighwayLayer(inputs):
    H = slim.conv2d(inputs, 64, [3,3])
    T = slim.conv2d(inputs, 64, [3,3], biases_initializer = tf.constant_initializer(-1.0), activation_fn=tf.nn.sigmoid)    
    outputs = H * T + inputs * (1.0 - T)
    return outputs   

def neural_networks():
    # 输入：训练的数量，一张图片的宽度，一张图片的高度 [-1,-1,16]
    inputs = tf.placeholder(tf.float32, [None, None, image_height], name="inputs")
    labels = tf.placeholder(tf.float32, [None, None, image_height], name="labels")
    
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    drop_prob = 1 - keep_prob

    shape = tf.shape(inputs)
    batch_size, image_width = shape[0], shape[1]

    layer = tf.reshape(inputs, [batch_size,image_width,image_height,1])

    layer = slim.conv2d(layer, 64, [3,3], normalizer_fn=slim.batch_norm)
    for i in range(POOL_COUNT):
        for j in range(5):
            layer = addHighwayLayer(layer)
        layer = tf.layers.dropout(layer, drop_prob)
        layer = slim.conv2d(layer, 64, [3,3], stride=[2, 2], normalizer_fn=slim.batch_norm)  

    predictions = slim.conv2d(layer, POOL_SIZE*POOL_SIZE, [3,3], normalizer_fn=slim.batch_norm, activation_fn=None)

    _predictions = tf.layers.flatten(predictions)
    _labels = tf.layers.flatten(labels)
    loss = tf.reduce_mean(tf.square(_predictions - _labels))

    return inputs, labels, predictions, keep_prob, loss

def http(url,param=None):
    if param !=None:
        paramurl = urllib.parse.urlencode(param)
        url = "%s?%s"%(url,paramurl)
        r = urllib.request.urlopen(url, timeout=30)
    else:    
        r = urllib.request.urlopen(url, timeout=30)
    return r.read()

r = http('http://192.168.2.113:8888/')
fonts = json.loads(r.decode('utf-8'))
ENGFontNames = fonts['eng']
print("EngFontNames", ENGFontNames)
CHIFontNames = fonts['chi']
print("CHIFontNames", CHIFontNames)
AllFontNames = ENGFontNames + CHIFontNames

def getRedomText(CHARS, word_dict, font_length):
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

def getImage(text, font_name, font_length, font_size, noise=False, fontmode=None, fonthint=None):
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
    _img = Image.open(io.BytesIO(r))
    img=Image.new("RGB",_img.size,(255,255,255))
    img.paste(_img,(0,0),_img)
    img = utils.trim(img)
    
    if noise:
        w,h = img.size
        _h = random.randint(9, image_height)
        _w = round(w * _h / h)
        img = img.resize((_w,_h), Image.ANTIALIAS)
        img = np.asarray(img)
        img = 1 - utils.img2gray(img)/255.   
        img = utils.dropZeroEdges(img)

        filter = np.random.random(img.shape) - 0.9
        filter = np.maximum(filter, 0) 
        img = img + filter * 5
        imin, imax = img.min(), img.max()
        img = (img - imin)/(imax - imin)
    else:
        img = np.asarray(img)
        img = utils.img2gray(img) 
        img = utils.img2bwinv(img)
        img = img / 255.
        img = utils.dropZeroEdges(img)
    return img

eng_world_list = open(os.path.join(curr_dir,"eng.wordlist.txt"),encoding="UTF-8").readlines() 
# 生成一个训练batch ,每一个批次采用最大图片宽度
def get_next_batch(batch_size=128):
    images = []   
    to_images = []
    max_width_image = 0
    font_min_length = random.randint(10, 20)
    for i in range(batch_size):
        font_name = random.choice(AllFontNames)
        font_length = random.randint(font_min_length-5, font_min_length+5)
        font_size = random.randint(9, 64)  
        font_mode = random.choice([0,1,2,4])    
        text = getRedomText(CHARS, eng_world_list, font_length)          
        image= getImage(text, font_name, font_length, font_size, noise = True, fontmode = font_mode )
        image=utils.resize(image, height=image_height)
        images.append(image)

        to_image=getImage(text, font_name, font_length, image_height, noise = False, fontmode = font_mode, fonthint = 1)
        to_image=utils.resize(to_image, height=image_height)
        to_images.append(to_image)

        if image.shape[1] > max_width_image: 
            max_width_image = image.shape[1]
        if to_image.shape[1] > max_width_image: 
            max_width_image = to_image.shape[1]      

    max_width_image = max_width_image + (POOL_SIZE - max_width_image % POOL_SIZE)
    inputs = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(images)):
        image_vec = utils.img2vec(images[i], height=image_height, width=max_width_image, flatten=False)
        inputs[i,:] = np.transpose(image_vec)

    labels = np.zeros([batch_size, max_width_image, image_height])
    for i in range(len(to_images)):
        image_vec = utils.img2vec(to_images[i], height=image_height, width=max_width_image, flatten=False)
        labels[i,:] = np.transpose(image_vec)
    return inputs, labels

def train():
    global_step = tf.Variable(0, trainable=False)
    inputs, labels, predictions, keep_prob, loss = neural_networks()

    curr_dir = os.path.dirname(__file__)
    model_dir = os.path.join(curr_dir, "model_font2font_highway")
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    saver_prefix = os.path.join(model_dir, "model.ckpt")        

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_INITIAL)
    train_op = optimizer.minimize(loss, global_step=global_step)
   
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver = tf.train.Saver(max_to_keep=5)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore Model ...")
            saver.restore(session, ckpt.model_checkpoint_path)    
        while True:
            for batch in range(BATCHES):
                start = time.time()                
                train_inputs, train_labels = get_next_batch(BATCH_SIZE)             
                feed = {inputs: train_inputs, labels: train_labels, keep_prob: 0.95}
                b_loss, b_labels, b_predictions,  steps,  _ = \
                    session.run([loss, labels, predictions, global_step, train_op], feed)

                seconds = round(time.time() - start,2)
                print("step:", steps, "cost:", b_loss, "batch seconds:", seconds)
                if np.isnan(b_loss) or np.isinf(b_loss):
                    print("Error: cost is nan or inf")
                    train_labels_list = decode_sparse_tensor(train_labels)
                    for i, train_label in enumerate(train_labels_list):
                        print(i,list_to_chars(train_label))
                    return   
                
                if seconds > 60: 
                    print('Exit for long time')
                    return

                if steps > 0 and steps % REPORT_STEPS == 0:
                    test_inputs, test_labels = get_next_batch(1)             
                    feed = {inputs: test_inputs, labels: test_labels, keep_prob: 1}
                    b_predictions = session.run([predictions], feed)                     
                    b_predictions = np.reshape(b_predictions[0],test_labels[0].shape)    
                    #utils.pltshow(test_inputs[0])   
                   # utils.pltshow(test_labels[0])  
                    #utils.pltshow(b_predictions)             
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_input.png"%steps), test_inputs[0]*255)
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_label.png"%steps), test_labels[0]*255)
                    cv2.imwrite(os.path.join(curr_dir,"test","%s_pred.png"%steps), b_predictions*255)


            saver.save(session, saver_prefix, global_step=steps)

if __name__ == '__main__':
    train()