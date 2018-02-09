# coding=utf-8
#!/sbin/python

import numpy as np
import paddle.v2 as paddle
import json
import os
import random
import sys
import time
import shutil 
import logging
import gc
import commands, re  
import threading

# home = "/home/kesci/work/"
# data_path = "/mnt/BROAD-datasets/video/"
# param_file = "/home/kesci/work/param2.data"
# param_file_bak = "/home/kesci/work/param2.data.bak"
# result_json_file = "/home/kesci/work/ai2.json"

channels_num = 4   # 图片先分层
class_dim = 2 # 分类 0，背景， 1，精彩
train_size = 8 # 学习的关键帧长度
buf_size = 10240
batch_size = 2048//train_size

home = os.path.dirname(__file__)
data_path = os.path.join(home,"data")
model_path = os.path.join(home,"model")
param_file = os.path.join(model_path,"param_%sx%s.tar"%(train_size,channels_num))
result_json_file = os.path.join(model_path,"ai2.json")
out_dir = os.path.join(model_path, "out")
if not os.path.exists(model_path): os.mkdir(model_path)
if not os.path.exists(out_dir): os.mkdir(out_dir)
np.set_printoptions(threshold=np.inf)

def load_data(filter=None):
    data = json.loads(open(os.path.join(data_path,"meta.json")).read())
    training_data = []
    validation_data = []
    testing_data = []
    for data_id in data['database']:
        if filter!=None and data['database'][data_id]['subset']!=filter:
            continue
        if data['database'][data_id]['subset'] == 'training':
            if os.path.exists(os.path.join(data_path,"training", "%s.pkl"%data_id)):
                training_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
        elif data['database'][data_id]['subset'] == 'validation':
            if os.path.exists(os.path.join(data_path,"validation", "%s.pkl"%data_id)):
                validation_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
        elif data['database'][data_id]['subset'] == 'testing':
            if os.path.exists(os.path.join(data_path,"testing", "%s.pkl"%data_id)):
                testing_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
    print('load data train %s, valid %s, test %s'%(len(training_data), len(validation_data), len(testing_data)))
    return training_data, validation_data, testing_data

def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(), ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, act=active_type)
    
def shortcut(ipt, n_in, n_out, stride):
    if n_in != n_out:
        return conv_bn_layer(ipt, n_out, 1, stride, 0, paddle.activation.Linear())
    else:
        return ipt

def basicblock(ipt, ch_out, stride):
    ch_in = ch_out * 2
    tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Relu())

def layer_warp(block_func, ipt, features, count, stride):
    tmp = block_func(ipt, features, stride)
    for i in range(1, count):
        tmp = block_func(tmp, features, 1)
    return tmp

def resnet(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(ipt, ch_in=channels_num, ch_out=64, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 64, n, 2)
    res2 = layer_warp(basicblock, res1, 64, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    res4 = layer_warp(basicblock, res3, 64, n, 2)
    res5 = layer_warp(basicblock, res4, 64, n, 2)
    res6 = layer_warp(basicblock, res5, 64, n, 2)
    return res6


def printLayer(layer):
    print("depth:",layer.depth,"height:",layer.height,"width:",layer.width,"num_filters:",layer.num_filters,"size:",layer.size,"outputs:",layer.outputs)

def network(drop=True):
    # 每批32张图片，将输入转为 1 * 256 * 256 CHW 
    x = paddle.layer.data(name='x', height=64, width=64, type=paddle.data_type.dense_vector(train_size*2048))

    # 是否精彩分类
    a = paddle.layer.data(name='a', type=paddle.data_type.integer_value(class_dim))

    net = resnet(x, 20)
    net = paddle.layer.dropout(input=net, dropout_rate=0.5)
    net_class_fc = paddle.layer.fc(input=net, size=class_dim, act=paddle.activation.Softmax())
    costs = paddle.layer.classification_cost(input=net_class_fc, label=a)
    
    parameters = paddle.parameters.create(costs)
    adam_optimizer = paddle.optimizer.Adam(learning_rate=1e-3,
        learning_rate_schedule="pass_manual", learning_rate_args="1:1.0,2:0.9,3:0.8,4:0.7,5:0.6,6:0.5",)
    return costs, parameters, adam_optimizer, net_class_fc

def read_data(v_data):
    batch_data = np.zeros((train_size, 2048))  
    w = v_data.shape[0]
    for i in range(w):
        _data = np.reshape(v_data[i], (1, 2048))
        batch_data = np.append(batch_data[1:, :], _data, axis=0)
        if random.random()<2./train_size: continue
        if i>=train_size:
            fix_batch_data = np.reshape(batch_data,(channels_num, train_size, 2048//channels_num))
            yield i, np.ravel(fix_batch_data)

data_pool_0 = []    #负样本
data_pool_1 = []    #正样本
training_data, validation_data, _ = load_data()
def readDatatoPool():
    size = len(training_data)+len(validation_data)
    c = 0
    
    for i in range(size):
        if i%2==0:
        # if True:
            data = random.choice(training_data)
            v_data = np.load(os.path.join(data_path,"training", "%s.pkl"%data["id"]))               
        else:
            data = random.choice(validation_data)
            v_data = np.load(os.path.join(data_path,"validation", "%s.pkl"%data["id"]))               

        # print "reading", data["id"], v_data.shape , len(data_pool_0), len(data_pool_1)
        label = np.zeros([v_data.shape[0]], dtype=np.int)
        for annotations in data["data"]:
            segment = annotations['segment']
            for i in range(int(segment[0]),int(segment[1]+1)):
                label[i] = 1

        for i, _data in read_data(v_data): 
            out_a = label[i-train_size:i]         
            if sum(out_a) == train_size :                   
                data_pool_0.append((_data, 1))
            elif sum(out_a) == 0:
                data_pool_1.append((_data, 0))
        while len(data_pool_1)>buf_size:
            # print("r")
            time.sleep(1) 
                
def reader_get_image_and_label():
    def reader():
        t1 = threading.Thread(target=readDatatoPool, args=())
        t1.start()
        while t1.isAlive():
            while len(data_pool_1)==0:
                # print("wait", len(data_pool_0), len(data_pool_1))
                time.sleep(1)
            if random.random()>0.5 and len(data_pool_0)>0:
                data_pool = data_pool_0
                v = 1.0*len(data_pool_0)/len(data_pool_1)
            else:
                data_pool = data_pool_1
                v = 1.0*len(data_pool_1)/len(data_pool_0)
            if random.random()>v:
                x, a = random.choice(data_pool)
            else:    
                x, a = data_pool.pop(random.randrange(len(data_pool)))
            yield x, a
    return reader

status ={}
status["starttime"]=time.time()
status["steptime"]=time.time()
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id>0 and event.batch_id % 10 == 0:
            print("Time %.2f, Pass %d, Batch %d, Cost %f, %s (%s/%s)" % (
                time.time() - status["steptime"], event.pass_id, event.batch_id, event.cost, event.metrics,
                len(data_pool_0), len(data_pool_1)) )
            status["steptime"]=time.time()
            with open(param_file, 'wb') as f:
                paddle_parameters.to_tar(f)

# for i in range(train_size):
#     print(i,get_box_point(i)) 

if __name__ == '__main__':
    print("paddle init ...")
    # paddle.init(use_gpu=False, trainer_count=2) 
    paddle.init(use_gpu=True, trainer_count=1)
    print("get network ...")
    cost, paddle_parameters, adam_optimizer, _ = network()
    print('set reader ...')
    train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
    feeding={'x':0, 'a':1}
    # feeding={'x':0, 'a':1} 
    if os.path.exists(param_file):
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(param_file)
        print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
        print("loading parameters %s ..."%param_file)
        paddle_parameters = paddle.parameters.Parameters.from_tar(open(param_file,"rb"))

    trainer = paddle.trainer.SGD(cost=cost, parameters=paddle_parameters, update_equation=adam_optimizer)
    print("start train ...")
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=8)
    print("paid:", time.time() - status["starttime"])