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

class_dim = 2 # 分类 1，背景， 2，精彩
anchors_dim = 5 # 窗口的大小，（0.75，0.625，0.5，0.25，0.125）
train_size = 32 # 学习的关键帧长度
anchors_num = 16*32 + 8*16 + 4*8 + 2*4 + 1*2  
buf_size = 8192
batch_size = 128

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

def cnn2(input,filter_size,num_channels,num_filters=64, padding=1):
    net = paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels,
         num_filters=num_filters, stride=1, padding=padding, act=paddle.activation.Linear())
    net = paddle.layer.batch_norm(input=net, act=paddle.activation.Relu())
    return paddle.layer.img_pool(input=net, pool_size=2, pool_size_y=1, stride=2, stride_y=1, pool_type=paddle.pooling.Max())

def cnn1(input,filter_size,num_channels,num_filters=64, padding=1, act=paddle.activation.Linear()):
    return  paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels,
         num_filters=num_filters, stride=1, padding=padding, act=act)

def network():
    # 每批32张图片，将输入转为 1 * 256 * 256 CHW 
    x = paddle.layer.data(name='x', height=1, width=2048, type=paddle.data_type.dense_vector(2048*train_size))  

    c_1024 = paddle.layer.data(name='c_1024', type=paddle.data_type.dense_vector(1024*train_size*class_dim))
    c_512 = paddle.layer.data(name='c_512', type=paddle.data_type.dense_vector(512*train_size*class_dim))
    c_256 = paddle.layer.data(name='c_256', type=paddle.data_type.dense_vector(256*train_size*class_dim))
    c_128 = paddle.layer.data(name='c_128', type=paddle.data_type.dense_vector(128*train_size*class_dim))
    c_64 = paddle.layer.data(name='c_64', type=paddle.data_type.dense_vector(64*train_size*class_dim))

    b_1024 = paddle.layer.data(name='b_1024', type=paddle.data_type.dense_vector(1024*2))
    b_512 = paddle.layer.data(name='b_512', type=paddle.data_type.dense_vector(512*2))
    b_256 = paddle.layer.data(name='b_256', type=paddle.data_type.dense_vector(256*2))
    b_128 = paddle.layer.data(name='b_128', type=paddle.data_type.dense_vector(128*2))
    b_64 = paddle.layer.data(name='b_64', type=paddle.data_type.dense_vector(64*2))

    c = [c_1024, c_512, c_256, c_128, c_64]
    b = [b_1024, b_512, b_256, b_128, b_64]

    main_nets = []
    net = cnn2(x,  3,  train_size, 64, 1)
    main_nets.append(net)
    net = cnn2(net, 3, 64, 64, 1)
    main_nets.append(net)
    net = cnn2(net,  3,  64, 64, 1)
    main_nets.append(net)
    net = cnn2(net,  3,  64, 64, 1)
    main_nets.append(net)
    net = cnn2(net,  3,  64, 64, 1)
    main_nets.append(net)
  
    # 分类网络
    nets_class = []
    # box网络
    nets_box = []

    for i  in range(len(main_nets)):
        net = cnn1(main_nets[i], 3, 64 ,anchors_dim*class_dim, 1, act=paddle.activation.Softmax())
        nets_class.append(net)
        net = cnn1(main_nets[i], 3, 64 ,2, 1)
        nets_box.append(net)
        
    costs =[]
    for i in range(len(main_nets)):
        net_cost = paddle.layer.classification_cost(input=nets_class[i], label=c[i])
        box_cost = paddle.layer.square_error_cost(input=nets_box[i], label=b[i])
        costs += [net_cost, box_cost]

    parameters = paddle.parameters.create(costs)
    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.1/batch_size)
    return costs, parameters, adam_optimizer, (nets_class, nets_box) 


data_pool = []
training_data, validation_data, _ = load_data("")
def readDatatoPool():
    size = len(training_data)+len(validation_data)
    c = 0
    for i in range(size):
        print(i)
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
        if event.batch_id>0 and event.batch_id % 10 == 0:
            print("Pass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics) )
            with open(param_file, 'wb') as f:
                paddle_parameters.to_tar(f)
        else:
            print(".")
print("paddle init ...")
paddle.init(use_gpu=False, trainer_count=2) 
# paddle.init(use_gpu=True, trainer_count=1)
print("get network ...")
cost, paddle_parameters, adam_optimizer, output = network()
print('set reader ...')
train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
# train_reader = paddle.batch(reader_get_image_and_label(True), batch_size=64)
feeding={'x': 0, 'y': 1}
 
trainer = paddle.trainer.SGD(cost=cost, parameters=paddle_parameters, update_equation=adam_optimizer)
print("start train ...")
trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=8)
