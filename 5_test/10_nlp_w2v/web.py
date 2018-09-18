#!/usr/bin/python3
'''
集成WEB服务
gunicorn -b 0.0.0.0:8080 web:app
'''
from flask import Flask, request, Response
import numpy as np
import tensorflow as tf
import os
from declare import networks 
from utils import splitSentence, getWord2Vec, clearData, get_word_dict, get_word_embed
import json
import time

def init():
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

    out_dir = os.path.dirname(__file__)
    model_dir = os.path.join(out_dir, "model", "declare")
    if not os.path.exists(model_dir):
        raise Exception("error: can't dind model dir")

    inputs_declare, inputs_declare_seq_len, \
        inputs_declare_other, inputs_declare_other_seq_len, \
        inputs_hsmodel, inputs_hsmodel_seq_len, \
        labels, prediction, optimizer, cost, accuracy = networks()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restore Model", ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)    
    else:
        raise Exception("error: can't load checkpoint data")

    return session, inputs_declare, inputs_declare_seq_len, \
            inputs_declare_other, inputs_declare_other_seq_len, \
            inputs_hsmodel, inputs_hsmodel_seq_len, prediction

session, inputs_declare, inputs_declare_seq_len, \
    inputs_declare_other, inputs_declare_other_seq_len, \
    inputs_hsmodel, inputs_hsmodel_seq_len, prediction = init()

# preds_declare 需要预测的文字
# preds_hsmodel 海关规范申报栏位
def generate_batch(preds_hsmodel, preds_declare):

    batch_size = len(preds_declare)
    MAX_HSMODEL_LEN = 20

    if len(preds_hsmodel)> MAX_HSMODEL_LEN:
        raise Exception("规范申报的项目个数超过20个")

    inputs_declare = [[] for i in range(batch_size)]
    inputs_declare_other = [[] for i in range(batch_size)]
    inputs_hsmodel = [[] for i in range(batch_size)]
    inputs_declare_seq_len = np.zeros(batch_size, dtype=np.int32)
    inputs_declare_other_seq_len = np.zeros(batch_size, dtype=np.int32)
    inputs_hsmodel_seq_len = np.zeros((batch_size, MAX_HSMODEL_LEN), dtype=np.int32)

    # 先整理 preds_hsmodel 海关规范申报栏位，不会变化
    hsmodel_data=[]
    hsmodel_seq_len_data = np.zeros(MAX_HSMODEL_LEN, dtype=np.int32)
    for i, hsmodel in enumerate(preds_hsmodel):
        hsmodel_data.append([])
        # print("hsmodel", hsmodel)
        for w in splitSentence(hsmodel):
            # print(w)
            hsmodel_data[i].append(getWord2Vec(w))
        hsmodel_seq_len_data[i]=len(hsmodel_data[i])

    # 每一批都一样    
    for i in range(batch_size):
        inputs_hsmodel[i] = hsmodel_data
        inputs_hsmodel_seq_len[i] = hsmodel_seq_len_data

    # 再整理 preds_declare 需要预测的列表
    for i, declare in enumerate(preds_declare):
        # print("declare", declare)
        for w in splitSentence(declare):
            # print(w)
            inputs_declare[i].append(getWord2Vec(w))
        inputs_declare_seq_len[i] = len(inputs_declare[i])

        # 输出同批需要预测的其它栏位
        declare_list_other_indexs = list(range(len(preds_declare)))
        declare_list_other_indexs.remove(i)
        for j in declare_list_other_indexs:
            for w in splitSentence(preds_declare[j]):
                inputs_declare_other[i].append(getWord2Vec(w))
        inputs_declare_other_seq_len[i] = len(inputs_declare_other[i])    

    inputs_declare_vec = np.zeros(shape=(batch_size, max(inputs_declare_seq_len),  256), dtype=np.float32)
    inputs_declare_other_vec = np.zeros(shape=(batch_size, max(inputs_declare_other_seq_len),  256), dtype=np.float32)
    inputs_hsmodel_vec = np.zeros(shape=(batch_size, MAX_HSMODEL_LEN, max(inputs_hsmodel_seq_len.flatten()), 256), dtype=np.float32)
    for i in range(batch_size):
        for j,v in enumerate(inputs_declare[i]): 
            inputs_declare_vec[i][j] = v
        for j,v in enumerate(inputs_declare_other[i]): 
            inputs_declare_other_vec[i][j] = v
        for k, inputs_hsmodel_item in enumerate(inputs_hsmodel[i]): 
            for l, v in enumerate(inputs_hsmodel_item):
                inputs_hsmodel_vec[i][k][l] = v

    return inputs_declare_vec, inputs_declare_seq_len, \
            inputs_declare_other_vec, inputs_declare_other_seq_len, \
            inputs_hsmodel_vec, inputs_hsmodel_seq_len

def pred(text):
    # start = time.time() 
    objs = json.loads(text)

    # 清洗数据
    hsmodel_list, declare_list = clearData(objs[0], objs[1])
    # print(u"接收数据", time.time()-start)

    result={}
    result['result'] = {}
    fix=0
    while len(declare_list) > 0 and len(hsmodel_list) > 0:
        # start = time.time() 

        preds_declare, preds_declare_seq_len, \
            preds_declare_other, preds_declare_other_seq_len, \
            preds_hsmodel, preds_hsmodel_seq_len = generate_batch(hsmodel_list, declare_list)
        # print(u"第 %s 次分词，生成向量数据"%fix, time.time()-start)

        # start = time.time() 
        y = session.run(prediction, feed_dict={inputs_declare: preds_declare, 
                                                inputs_declare_seq_len: preds_declare_seq_len,
                                                inputs_declare_other:preds_declare_other, 
                                                inputs_declare_other_seq_len: preds_declare_other_seq_len,
                                                inputs_hsmodel: preds_hsmodel,
                                                inputs_hsmodel_seq_len: preds_hsmodel_seq_len
                                                })

        # print(u"第 %s 次预测产生结果"%fix, time.time()-start)
        # start = time.time() 

        for _ in range(fix):
            y = y * 0.5
 
        pred_list = y.tolist()
        firsts={}
        for i, declare in enumerate(declare_list):
            max_value = max(pred_list[i])
            max_index = pred_list[i].index(max_value)

            # 如果预测的值界外 
            if max_index >= len(hsmodel_list):
                if len(declare_list)>1:
                    continue
                else:
                    # 如果只剩下最后一个，无论如何也要找到一个存在的
                    while max_index>=len(hsmodel_list):
                        pred_list[i][max_index]=-1
                        max_value = max(pred_list[i])
                        max_index = pred_list[i].index(max_value)

            hsmodel = hsmodel_list[max_index]
            if hsmodel in firsts:
                if max_value> firsts[hsmodel][1]:
                    firsts[hsmodel]=[declare, max_value]
            else:
                firsts[hsmodel]=[declare, max_value]

        for hsmodel in firsts: 
            declare = firsts[hsmodel][0]
            value = firsts[hsmodel][1]
            result['result'][declare] = (hsmodel, value)
            hsmodel_list.remove(hsmodel)
            declare_list.remove(declare)      

        # print(u"第 %s 次整理数据，输出结果"%fix, time.time()-start)
        fix += 1
    return json.dumps(result, ensure_ascii=False)

import jieba
jieba.initialize()
get_word_dict()
get_word_embed()

app = Flask(__name__)

@app.route('/')
def index():
	return '''<!DOCTYPE html>
        <html lang="zh-CN">
        <body>
            <form action="/pred" method="post" enctype="multipart/form-data">
                输入预测的JSON数据: <br>
                分为两个list，前面是规范申报的栏位，长度20，不足会自动补全；栏位前面的序号可写可不写；<br>
                后面是需要预测的值，长度不限，前后的长度不需要保持对应一致<br>
                返回的结果为值对应栏位的概率，其中概率为20个长度，合计为1，最大值的索引为最有可能的栏位分类。<br>
                返回格式： [[...20项的预测，合计为1...],...输入的待预测值个数...,[...20项的预测，合计为1...]]<br><br>
                <textarea rows="20" cols="100" name="pred" id="pred">%s</textarea><br>
                <button type="submit" class="btn btn-default">Submit</button>
            </form><br><br>
            注意，请关注默认案例中最后一项的概率，其预测的最大值索引和第一项是同一个都是用途，但得分没有第一个的高。
        </body>
        </html>
    '''%'''
    [
        ["1品牌类型","2出口享惠情况","3用途","4外观","5是否成卷","6是否单面自粘","7成分含量","8规格尺寸","9若为半导体晶圆制造用","10品牌","11型号","12GTIN","13CAS","14其他"],
        ["自粘塑料片/厂家:XXXXXX","无型号","料号:XXXXXX","规格尺寸:XXXX*XXX*XXXXmm","成分含量:PET90%丙烯酸聚合物10%","片状","包装不成卷","单面自粘","非半导体晶圆制"]
    ]'''

@app.route('/pred', methods=['POST'])
def single_digit():
    file = request.form['pred']
    if file :
    	return pred(file)
    else:
        return 'No file upload'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
