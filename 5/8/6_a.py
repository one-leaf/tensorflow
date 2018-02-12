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

class_dim = 2 # 分类 0，背景， 1，精彩
train_size = 64 # 学习的关键帧长度
block_size = 8

buf_size = 4096
batch_size = 2048//(train_size*block_size)

home = os.path.dirname(__file__)
data_path = os.path.join(home,"data")
model_path = os.path.join(home,"model")
cls_param_file = os.path.join(model_path,"param_cls.tar")
box_param_file = os.path.join(model_path,"param_box.tar")

result_json_file = os.path.join(model_path,"ai2.json")
out_dir = os.path.join(model_path, "out")
if not os.path.exists(model_path): os.mkdir(model_path)
if not os.path.exists(out_dir): os.mkdir(out_dir)
np.set_printoptions(threshold=np.inf)

training_path = os.path.join(data_path,"training","image_resnet50_feature")
validation_path = os.path.join(data_path,"validation","image_resnet50_feature")
testing_path = os.path.join(data_path,"testing","image_resnet50_feature")

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

def resnet(ipt, depth=32, drop=False):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(ipt, ch_in=2048*block_size//64//64, ch_out=64, filter_size=3, stride=1, padding=1)
    if drop:
        conv1 = paddle.layer.dropout(input=conv1, dropout_rate=0.5)
    res1 = layer_warp(basicblock, conv1, 64, n, 2)
    if drop:
        res1 = paddle.layer.dropout(input=res1, dropout_rate=0.5)
    res2 = layer_warp(basicblock, res1, 64, n, 2)
    if drop:
        res2 = paddle.layer.dropout(input=res2, dropout_rate=0.5)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    if drop:    
        res3 = paddle.layer.dropout(input=res3, dropout_rate=0.5)
    res4 = layer_warp(basicblock, res3, 64, n, 2)
    res5 = layer_warp(basicblock, res4, 64, n, 2)
    res6 = layer_warp(basicblock, res5, 64, n, 2)
    return res6

def printLayer(layer):
    print("depth:",layer.depth,"height:",layer.height,"width:",layer.width,"num_filters:",layer.num_filters,"size:",layer.size,"outputs:",layer.outputs)


def cnn(input,filter_size,num_channels,num_filters=64, stride=2, padding=1):
    return paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels, 
        num_filters=num_filters, stride=stride, padding=padding, act=paddle.activation.Relu())


def network(drop=True):
    # 每批32张图片，将输入转为 1 * 256 * 256 CHW 
    x = paddle.layer.data(name='x', height=64, width=64, type=paddle.data_type.dense_vector_sequence(2048*block_size))   

    # 是否精彩分类
    a = paddle.layer.data(name='a', type=paddle.data_type.integer_value_sequence(class_dim))

#    net = resnet(x, 32, drop)

    net = cnn(x,    8, 2048*block_size//64//64, 64, 2, 3)
    if drop:
        net = paddle.layer.dropout(input=net, dropout_rate=0.5)
    net = cnn(net,  6, 64, 64, 2, 2)
    if drop:
        net = paddle.layer.dropout(input=net, dropout_rate=0.5)    
    net = cnn(net,  4, 64, 64, 2, 1)
    if drop:
        net = paddle.layer.dropout(input=net, dropout_rate=0.5)            
    net = cnn(net,  3, 64, 64, 2, 1)
    # if drop:
    #     net = paddle.layer.dropout(input=net, dropout_rate=0.5)    
    net = cnn(net,  3, 64, 64, 2, 1)
 #   net = cnn(net,  3, 64, 64, 2, 1)


    # 当前图片精彩或非精彩分类
    net_class_gru = paddle.networks.simple_gru(input=net, size=128, act=paddle.activation.Tanh())
    net_class_fc = paddle.layer.fc(input=net_class_gru, size=class_dim, act=paddle.activation.Softmax())
    cost_class = paddle.layer.classification_cost(input=net_class_fc, label=a)
   
    adam_optimizer = paddle.optimizer.Adam(learning_rate=1e-3,
        learning_rate_schedule="pass_manual", learning_rate_args="1:1.0,2:0.9,3:0.8,4:0.7,5:0.6,6:0.5",)
    return cost_class, adam_optimizer, net_class_fc

# 读取精彩和非精彩, 离散数据
def read_data_cls(v_data, label): 
    w = v_data.shape[0]
    for i in range(block_size, w):
        _avg = 1.*sum(label[i-block_size:i])/block_size
        # 只要全部是精彩或全部是非精彩的片段
        if (_avg == 1 or _avg == 0) and random.random()>1./2:
            _data = np.stack([v_data[j] for j in range(i-block_size,i)])
            yield i, _data, int(_avg)

data_pool_0 = []    #负样本
data_pool_1 = []    #正样本
training_data, validation_data, _ = load_data()
def readDatatoPool():
    size = len(training_data)+len(validation_data)
    c = 0
    for i in range(size):
        if random.random()>1.0*len(validation_data)/size:
            data = random.choice(training_data)
            v_data = np.load(os.path.join(data_path,"training", "%s.pkl"%data["id"]))               
        else:
            data = random.choice(validation_data)
            v_data = np.load(os.path.join(data_path,"validation", "%s.pkl"%data["id"]))               

        # print "reading", data["id"], v_data.shape , len(data_pool_0), len(data_pool_1)
        w = v_data.shape[0]
        label = [0 for _ in range(w)]
        for annotations in data["data"]:
            segment = annotations['segment']
            for i in range(int(segment[0]),min(w,int(segment[1]+1))):
                label[i] = 1
        for i, _data, _label in read_data_cls(v_data, label):
            if _label==0:
                data_pool_0.append((_data, _label))
            else:
                data_pool_1.append((_data, _label))

        while len(data_pool_1)>buf_size:
            # print("r", len(data_pool_0), len(data_pool_1))
            time.sleep(1) 
              
def reader_get_image_and_label():
    def reader():
        t1 = threading.Thread(target=readDatatoPool, args=())
        t1.start()
        while t1.isAlive():            
            datas=[]
            labels=[]
            while (len(datas)<train_size):
                while len(data_pool_1)==0 or len(data_pool_0)==0:
                    print("w", len(data_pool_0), len(data_pool_1))
                    time.sleep(1)
                if random.random()>0.5 and len(data_pool_0)>0:
                    data_pool = data_pool_0
                    v = 1.0*len(data_pool_0)/len(data_pool_1)
                else:
                    data_pool = data_pool_1
                    v = 1.0*len(data_pool_1)/len(data_pool_0)
                if random.random()>v:
                    d, a = random.choice(data_pool)
                else:    
                    d, a = data_pool.pop(random.randrange(len(data_pool)))
                datas.append(d)
                labels.append(a)
            yield datas, labels   
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
            # cls_parameters.to_tar(open(cls_param_file, 'wb'))

def train():
    print("get network ...")
    cost, adam_optimizer, net_class_fc = network(True)

    # if os.path.exists(cls_param_file):
    #     (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(cls_param_file)
    #     print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
    #     print("loading cls parameters %s ..."%cls_param_file)
    #     cls_parameters = paddle.parameters.Parameters.from_tar(open(cls_param_file,"rb"))
    # else:
    cls_parameters = paddle.parameters.create(cost)

    print('set reader ...')
    train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
    feeding_class={'x':0, 'a':1} 

    trainer = paddle.trainer.SGD(cost=cost, parameters=cls_parameters, update_equation=adam_optimizer)
    print("start train class ...")
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding_class, num_passes=1)
    print("paid:", time.time() - status["starttime"])
    cls_parameters.to_tar(open(cls_param_file, 'wb'))

def infer():
    cost, adam_optimizer, net_class_fc = network(False) 
    cls_parameters = paddle.parameters.Parameters.from_tar(open(cls_param_file,"rb"))    
    inferer = paddle.inference.Inference(output_layer=net_class_fc, parameters=cls_parameters)

    for t in range(2):
        if t == 0:
            pdata = training_data
            ppath = "training"
        else:
            pdata = validation_data
            ppath = "validation"
        for data in pdata:
            filename = "%s.pkl"%data["id"]
            save_file = os.path.join(out_dir,filename)
            if os.path.exists(save_file): continue            
            v_data = np.load(os.path.join(data_path, ppath, filename))
            w = v_data.shape[0]
            values = np.zeros(w,dtype=np.float)
            for i in range(w):
                if i>=block_size:
                    _data = np.stack([v_data[j] for j in range(i-block_size,i)])
                else:
                    _null_data = [np.zeros((2048)) for j in range(block_size-i-1)]
                    _not_null_data = [v_data[j] for j in range(0,i+1)]
                    _data = np.stack(_null_data+_not_null_data)
                probs = inferer.infer(input=[([_data,],)])
                values[i] = probs[0][1]
            print("infered %s"%filename)
            np.save(open(save_file,"wb"), values)


if __name__ == '__main__':
    print("paddle init ...")
    # paddle.init(use_gpu=False, trainer_count=2) 
    paddle.init(use_gpu=True, trainer_count=1)
    train()
    # infer()