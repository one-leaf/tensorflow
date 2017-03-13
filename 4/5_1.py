# coding=utf-8
'''
集成WEB服务
'''
from flask import Flask, request, Response
import numpy as np
import tensorflow as tf
import os
from utils import img2gray, img2vec ,vec2text
from PIL import Image
import StringIO

def init():
    out_dir = os.path.dirname(__file__)
    log_dir = os.path.join(out_dir, "logs")
    if not os.path.exists(log_dir):
        raise Exception("error: can't dind log dir")

    checkpoint_prefix = os.path.join(log_dir, "model.ckpt")

    # 找到最新的运算模型文件
    metaFile= sorted(
        [
            (x, os.path.getctime(os.path.join(log_dir,x)))                  
            for x in os.listdir(log_dir) if x.endswith('.meta')  
        ],
        key=lambda i: i[1])[-1][0]

    sess = tf.Session()

    saver = tf.train.import_meta_graph(os.path.join(log_dir,metaFile))
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise Exception("error: can't load checkpoint data")

    x = tf.get_default_graph().get_tensor_by_name('x:0')
    prediction = tf.get_default_graph().get_tensor_by_name('prediction:0')
    return sess,prediction,x

sess,prediction,x = init()
char_set = "abcdefghijklmnopqrstuvwxyz0123456789"         # 验证码组成

def crack(file):
    img = Image.open(file.stream)
    image = np.array(img)
    image = img2vec(img2gray(image))
    y_ = sess.run([prediction], feed_dict={x: [image]})
    print(y_)
    result = "".join([char_set[s] for s in y_[0][0]])
    return result

app = Flask(__name__)

@app.route('/')
def index():
	return '''<!DOCTYPE html>
        <html lang="zh-CN">
        <body>
            <form action="/crack" method="post" enctype="multipart/form-data">
                Select image to upload: <br/><br/>
                <input type="file" name="file" id="file"> <br/><br/>
                <button type="submit" class="btn btn-default">Submit</button>
            </form><br/><br/>
        </body>
        </html>
    '''

@app.route('/crack', methods=['POST'])
def single_digit():
    file = request.files['file']
    if file :
    	return crack(file)
    else:
        return 'No file upload'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
