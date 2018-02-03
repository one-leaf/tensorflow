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

class_dim = 2 # 分类 0，背景， 1，精彩
box_dim = 2 # 偏移，左，右
train_size = 128 # 学习的关键帧长度
buf_size = 8192
batch_size = 1

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
    return paddle.layer.img_pool(input=net, pool_size=2, pool_size_y=2, stride=2, stride_y=2, pool_type=paddle.pooling.Max())

def cnn1(input,filter_size,num_channels,num_filters=64, stride=1, padding=1, act=paddle.activation.Linear()):
    return  paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels,
         num_filters=num_filters, stride=stride, padding=padding, act=act)

def printLayer(layer):
    print("depth:",layer.depth,"height:",layer.height,"width:",layer.width,"num_filters:",layer.num_filters,"size:",layer.size,"outputs:",layer.outputs)

def network():
    # 每批32张图片，将输入转为 1 * 256 * 256 CHW 
    x = paddle.layer.data(name='x', height=1, width=2048, type=paddle.data_type.dense_vector_sequence(2048))  
    x_emb = paddle.layer.embedding(input=x, size=train_size*2048)

    c = paddle.layer.data(name='c', type=paddle.data_type.integer_value_sequence(class_dim))
    c_emb = paddle.layer.embedding(input=c, size=train_size)

    b = paddle.layer.data(name='b', type=paddle.data_type.dense_vector_sequence(box_dim))
    b_emb = paddle.layer.embedding(input=b, size=train_size)

    main_nets = []
    net = cnn2(x_emb,  3,  1, train_size, 1)
    main_nets.append(net)
    net = cnn2(net, 3, train_size, train_size, 1)
    main_nets.append(net)
    net = cnn2(net,  3,  train_size, train_size, 1)
    main_nets.append(net)
    net = cnn2(net,  3,  train_size, train_size, 1)
    main_nets.append(net)
    net = cnn2(net,  3,  train_size, train_size, 1)
    main_nets.append(net)
  
    # 分类网络
    nets_class = []
    # box网络
    nets_box = []

    for i  in range(len(main_nets)):
        main_net = main_nets[i]
        printLayer(main_net)
        block_expand = paddle.layer.block_expand(input= main_net, num_channels=1, stride_x=1, stride_y=1, block_x=main_net.width, block_y=main_net.height)
        printLayer(block_expand)

        net = cnn1(block_expand, 3, 64, class_dim, 1, 0)
        nets_class.append(net)
        net = cnn1(block_expand, 3, 64, box_dim, 1, 0)
        nets_box.append(net)

    
    # net_class = paddle.layer.concat(input=nets_class)
    # gru_forward = paddle.networks.simple_gru(input=net, size=128, act=paddle.activation.Relu())

    net_class = paddle.layer.fc(input=nets_class, size=class_dim, act=paddle.activation.Softmax())
    net_cost = paddle.layer.classification_cost(input=net_class, label=c)
  
    # net_box = paddle.layer.concat(input=nets_box)
    # gru_forward = paddle.networks.simple_gru(input=net, size=128, act=paddle.activation.Relu())

    net_box = paddle.layer.fc(input=nets_box, size=box_dim, act=paddle.activation.Tanh())
    box_cost = paddle.layer.square_error_cost(input=net_box, label=b)
    costs = [net_cost + box_cost]

    parameters = paddle.parameters.create(costs)
    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    return costs, parameters, adam_optimizer, (nets_class, nets_box) 


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
            
        # batch_data = np.zeros((2048, train_size))    
        batch_data = [np.zeros(2048) for _ in range(train_size)]   
        w = v_data.shape[0]
        # label = np.zeros([w], dtype=np.int)

        # fix_segments =[]
        # for annotations in data["data"]:
        #     segment = annotations['segment']
        #     for i in range(int(segment[0]),int(segment[1]+1)):
        #         label[i] = 1

        for i in range(w):
            # _data = np.reshape(v_data[i], (2048,1))
            # batch_data = np.append(batch_data[:, 1:], _data, axis=1)
            batch_data.append(v_data[i])
            batch_data.pop(0)
            # if random.random()>0.9: continue
            if random.random() > 1.0/(train_size//2): continue
            fix_segments =[]
            for annotations in data["data"]:
                segment = annotations['segment']
                if segment[0]>i or segment[1]<i- train_size:
                    continue
                fix_segments.append([max(0,segment[0]-i-train_size),min(train_size,i-segment[1])])
                out_c, out_b = calc_value(fix_segments)
                # data_pool.append((np.ravel(batch_data), out_c, out_b))
                data_pool.append((batch_data, out_c, out_b))
        while len(data_pool)>buf_size:
            # print('r')
            time.sleep(0.1) 

# 计算 IOU,输入为 x1,x2 坐标
def calc_iou(src, dst):
    if src[1]<dst[0] or dst[1]<src[0]:
        return 0
    if dst[1]>src[1]:
        if dst[0]>src[0]:
            return (src[1]-dst[0])/(dst[1]-src[0])
        else:
            return (src[1]-src[0])/(dst[1]-dst[0])
    else:
        if src[0]>dst[0]:
            return (dst[1]-src[0])/(src[1]-dst[0])
        else:
            return (dst[1]-dst[0])/(src[1]-src[0])         
    
# 计算
def calc_value(segments):
    out_c=[0 for _ in range(train_size)]
    out_b=[np.zeros(2) for _ in range(train_size)]

    for i in range(train_size):
        src = (max(i-train_size//2,0),min(i+train_size//2,train_size))
        ious = []
        for dst in segments:
            ious.append(calc_iou(src, dst))
        max_ious = max(ious)
        max_ious_index = ious.index(max_ious)
        if max_ious>0.5:
            out_c[i]=1
            out_b[i][0]=(segments[max_ious_index][0]-src[0])/train_size
            out_b[i][1]=(segments[max_ious_index][1]-src[1])/train_size
        else:
            out_c[i]=0            
        
    return out_c, out_b
                
def reader_get_image_and_label():
    def reader():
        t1 = threading.Thread(target=readDatatoPool, args=())
        t1.start()
        while t1.isAlive():
            while len(data_pool)==0:
                # print('w')
                time.sleep(1)
            x , y, z = data_pool.pop(random.randrange(len(data_pool)))
            yield x, y, z
    return reader

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id>0 and event.batch_id % 10 == 0:
            print("Pass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics) )
            with open(param_file, 'wb') as f:
                paddle_parameters.to_tar(f)
        # else:
            # print(".")
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=2) 
paddle.init(use_gpu=True, trainer_count=2)
print("get network ...")
cost, paddle_parameters, adam_optimizer, output = network()
print('set reader ...')
train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
# train_reader = paddle.batch(reader_get_image_and_label(True), batch_size=64)
feeding={'x':0, 'c':1, 'b':2}
 
trainer = paddle.trainer.SGD(cost=cost, parameters=paddle_parameters, update_equation=adam_optimizer)
print("start train ...")
trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=8)
