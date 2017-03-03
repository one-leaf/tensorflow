# coding=utf-8
'''
集成WEB服务
'''
from flask import Flask, request
import numpy as np
import tensorflow as tf
import os


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

def crack(file):
    return "OK"

app = Flask(__name__)

@app.route('/')
def index():
	return '''<!DOCTYPE html><html>
        <body>
            <form action="/crack" method="post" enctype="multipart/form-data">
                Select image to upload:
                <input type="file" name="file" id="file">
                <input type="submit" value="Crack Image" name="submit">
            </form>
        </body>
        </html>
    '''

@app.route('/crack', methods=['POST'])
def single_digit():
    file = request.files['file']
    if file :
    	return crack(file)

if __name__ == '__main__':
	app.run()