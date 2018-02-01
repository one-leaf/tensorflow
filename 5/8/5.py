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


home = os.path.dirname(__file__)
data_path = os.path.join(home,"data")
model_path = os.path.join(home,"model")
param_file = os.path.join(model_path,"param2.tar")
result_json_file = os.path.join(model_path,"ai2.json")
out_dir = os.path.join(model_path, "out")

# home = "/home/kesci/work/"
# data_path = "/mnt/BROAD-datasets/video/"
# param_file = "/home/kesci/work/param2.data"
# param_file_bak = "/home/kesci/work/param2.data.bak"
# result_json_file = "/home/kesci/work/ai2.json"

if not os.path.exists(model_path): os.mkdir(model_path)
if not os.path.exists(out_dir): os.mkdir(out_dir)

class_dim = 2 # 0: 0  1:  0-->1 2: 1--->0
train_size = 16 # 学习的关键帧长度
buf_size = 8192

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

def cnn(input,filter_size,num_channels,num_filters=64, stride=2, padding=1):
    return paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels, 
        num_filters=num_filters, stride=stride, padding=padding, act=paddle.activation.Relu())

def network():
    # -1 ,2048*5 
    x = paddle.layer.data(name='x', width=2048, height=train_size, type=paddle.data_type.dense_vector(2048*train_size))
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(class_dim))
   
    net = cnn(x,   4,  1, 16, 2, 1)
    net = cnn(net, 4, 16, 32, 2, 1)
    net = cnn(net, 4, 32, 64, 2, 1)
    net = cnn(net, 4, 64, 64, 2, 1)
    # net = cnn(net, 4, 64, 64, 2, 1)
    output = paddle.layer.fc(input=net, size=class_dim, act=paddle.activation.Softmax())

    # sliced_feature = paddle.layer.block_expand(input=net, num_channels=64, stride_x=1, stride_y=1, block_x=128, block_y=1)
    # gru_forward = paddle.networks.simple_gru(input=sliced_feature, size=64, act=paddle.activation.Relu())
    # gru_backward = paddle.networks.simple_gru(input=sliced_feature, size=64, act=paddle.activation.Relu(), reverse=True)
    # output = paddle.layer.fc(input=[gru_forward, gru_backward], size=class_dim, act=paddle.activation.Softmax())
  
    cost = paddle.layer.classification_cost(input=output, label=y)
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    return cost, parameters, adam_optimizer, output

data_pool = []
training_data, validation_data, _ = load_data()
def readDatatoPool():
    size = len(training_data)+len(validation_data)
    c = 0
    for i in range(size):
        if i%2==0:
            data = random.choice(training_data)
            v_data = np.load(os.path.join(data_path,"training", "%s.pkl"%data["id"]))               
        else:
            data = random.choice(validation_data)
            v_data = np.load(os.path.join(data_path,"validation", "%s.pkl"%data["id"]))               
            
        batch_data = np.zeros((2048, train_size))    
        w = v_data.shape[0]
        label = np.zeros([w], dtype=np.int)

        for annotations in data["data"]:
            segment = annotations['segment']
            for i in range(int(segment[0]),int(segment[1]+1)):
                label[i] = 1

        for i in range(w):
            _data = np.reshape(v_data[i], (2048,1))
            batch_data = np.append(batch_data[:, 1:], _data, axis=1)
            if i > train_size and random.random() > 0.25: 
                s = sum(label[i-train_size+1:i+1]) 
                if c > 32 and s == train_size and random.random() > 0.25: continue                    
                if c < -32 and s == 0 and random.random() > 0.25: continue                    
                if s == train_size:
                    v = 1 
                    c += 1
                elif s == 0:
                    v = 0
                    c -= 1
                else:
                    continue 

                data_pool.append((np.ravel(batch_data), v))

        while len(data_pool)>buf_size:
            time.sleep(0.1) 
                    
def reader_get_image_and_label():
    def reader():
        t1 = threading.Thread(target=readDatatoPool, args=())
        t1.start()
        while t1.isAlive():
            while len(data_pool)==0:
                time.sleep(1)
            x , y = data_pool.pop(random.randrange(len(data_pool)))
            yield x, y
    return reader

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id>0 and event.batch_id % 20 == 0:
            print("Pass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics) )
            with open(param_file, 'wb') as f:
                paddle_parameters.to_tar(f)
        
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=1) 
paddle.init(use_gpu=True, trainer_count=2)
print("get network ...")
cost, paddle_parameters, adam_optimizer, output = network()
print('set reader ...')
train_reader = paddle.batch(reader_get_image_and_label(), batch_size=128)
# train_reader = paddle.batch(reader_get_image_and_label(True), batch_size=64)
feeding={'x': 0, 'y': 1}
 
trainer = paddle.trainer.SGD(cost=cost, parameters=paddle_parameters, update_equation=adam_optimizer)
print("start train ...")
trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=8)
