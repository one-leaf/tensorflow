# coding=utf-8
'''
集成WEB服务
gunicorn -b 0.0.0.0:8080 web_ascii:app
'''
from flask import Flask, request, Response
import numpy as np
import tensorflow as tf
import os, time
# from PIL import Image
import json
import ocr_ascii_nolstm as ocr
import utils, cv2
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def init():
#    logits, inputs, labels, seq_len, input_keep_prob = ocr.neural_networks()
    global_step = tf.Variable(0, trainable=False)
    curr_learning_rate = 1e-5
    learning_rate = tf.placeholder(tf.float32, shape=[])                                            
    logits, inputs, labels, seq_len, input_keep_prob = ocr.neural_networks()
    loss = tf.nn.ctc_loss(labels=labels,inputs=logits, sequence_length=seq_len)
    cost = tf.reduce_mean(loss, name="cost")
    grads_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = grads_optimizer.compute_gradients(loss)
    capped_grads_and_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads_and_vars]
    optimizer = grads_optimizer.apply_gradients(capped_grads_and_vars, global_step=global_step)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=10, merge_repeated=False)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # restore sess
    model_dir = os.path.join(curr_dir, "model-ascii-nolstm")
    saver_prefix = os.path.join(model_dir, "model.ckpt")        
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        print("Restore Model ...")
        saver.restore(sess, ckpt.model_checkpoint_path)    
    return sess, inputs, seq_len, input_keep_prob, decoded, log_prob

curr_dir = os.path.dirname(__file__)
session, inputs, seq_len, input_keep_prob, decoded, log_prob = init()

def scan(file):
    img_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
    image = cv2.imdecode(img_array,0)
    split_images = utils.splitImg(image)
    
    ocr_texts = []

    for i, split_image in enumerate(split_images):
        image = utils.img2bwinv(split_image)
        image = utils.clearImg(image)
        image = utils.dropZeroEdges(image)  
        image = utils.resize(image, ocr.image_height)
        utils.save(image,os.path.join(curr_dir,"test","%s.png"%i))
        maxImageWidth = image.shape[1]
        maxImageWidth = maxImageWidth + (4 - maxImageWidth % 4)
        image_vec = utils.img2vec(image,ocr.image_height,maxImageWidth)
        ocr_inputs = np.zeros([1, maxImageWidth, ocr.image_height])
        ocr_inputs[0,:] = np.transpose(image_vec.reshape((ocr.image_height,maxImageWidth)))         
        ocr_seq_len = np.ones(ocr_inputs.shape[0]) * maxImageWidth // 4
        feed = {inputs: ocr_inputs, seq_len: ocr_seq_len,  input_keep_prob: 1.0}
        start = time.time()
        decoded_list = session.run(decoded[0], feed)
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
                1. 只能是 bmp 或 png 图片格式<br/>
                2. 只能是 白底 黑字<br/>
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
    os.environ['CUDA_VISIBLE_DEVICES'] = ''        
    app.run(host='0.0.0.0',port=8080)
