# coding=utf-8
'''
集成WEB服务
gunicorn -b 0.0.0.0:8080 web_ascii:app
'''
from flask import Flask, request, Response

import numpy as np
import tensorflow as tf
import os, time
from PIL import Image
import json
import font_ascii_ocr as ocr
import utils, cv2

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

curr_dir = os.path.dirname(__file__)

def init():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'    

    inputs, labels, global_step, \
            res_loss, res_optim, seq_len, res_acc, res_decoded, \
            net_g = ocr.neural_networks()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    model_dir = os.path.join(curr_dir, "model_ascii_srgan")
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    model_G_dir = os.path.join(model_dir, "FG")
    model_R_dir = os.path.join(model_dir, "FR")

    r_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='RES'), sharded=True)
    g_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SRGAN_g'), sharded=True)

    ckpt = tf.train.get_checkpoint_state(model_G_dir)
    if ckpt and ckpt.model_checkpoint_path:           
        print("Restore Model G...")
        g_saver.restore(session, ckpt.model_checkpoint_path)   

    ckpt = tf.train.get_checkpoint_state(model_R_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restore Model R...")
        r_saver.restore(session, ckpt.model_checkpoint_path)    

    return session, inputs, seq_len, res_decoded, net_g

session, inputs, seq_len, res_decoded, net_g = init()

def scan(file):
    img = Image.open(file.stream)
    image = np.array(img)
    image = utils.img2gray(image)
    image = utils.clearImgGray(image)    
    split_images = utils.splitImg(image)
    
    ocr_texts = []

    for i, split_image in enumerate(split_images):
        # image = utils.img2bwinv(split_image)
        image = 255. - split_image
        # image = utils.dropZeroEdges(image)  
        image = utils.resize(image, ocr.image_height)
        utils.save(image,os.path.join(curr_dir,"test","%s.png"%i))
        image = image / 255.
        ocr_inputs = np.zeros([1, ocr.image_size, ocr.image_size])
        ocr_inputs[0,:] = utils.img2img(image,np.zeros([ocr.image_size, ocr.image_size]))
        
        ocr_seq_len = np.ones(1) * (ocr.image_size * ocr.image_size ) // (ocr.POOL_SIZE * ocr.POOL_SIZE)

        start = time.time()
        p_net_g = session.run(net_g, {inputs: ocr_inputs}) 
        p_net_g = np.squeeze(p_net_g, axis=2)
        decoded_list = session.run(res_decoded[0], {inputs: p_net_g, seq_len: ocr_seq_len}) 
        seconds = round(time.time() - start,2)
        print("filished ocr %s , paid %s seconds" % (i,seconds))
        detected_list = ocr.decode_sparse_tensor(decoded_list)            
        for detect_number in detected_list:
            ocr_texts.append(ocr.list_to_chars(detect_number))

    return ocr_texts

app = Flask(__name__)

@app.route('/')
def index():
    return '''<!DOCTYPE html>
        <html lang="zh-CN">
        <body>
            <form action="/ocr" method="post" enctype="multipart/form-data">
                选择一个文件上传来 OCR （支持全部Windows字体和各种字号）: <br/>
                1. 只能是英文数字，不能跨表格线<br/>
                <br/>
                <br/>
                <input type="file" name="file" id="file"> <br/><br/>
                <button type="submit" class="btn btn-default">Submit</button>
            </form><br/><br/>
        </body>
        </html>
    '''

@app.route('/ocr', methods=['POST'])
def single_digit():
    file = request.files['file']
    if file :
        code = scan(file)
        print(code)
        return json.dumps(code, ensure_ascii=False)
    else:
        return 'No file upload'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
