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

channels_num = 4   # 图片先分层
class_dim = 2 # 分类 0，背景， 1，精彩
box_dim = 2 # 偏移，左，右
train_size = 256 # 学习的关键帧长度
buf_size = 8192
batch_size = 2048//train_size
block_size = train_size//4
area_ratio = (1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0.125)


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

def cnn2(input,filter_size,num_channels, num_filters=64, stride=1, padding=1):
    net = paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels,
         num_filters=num_filters, stride=stride, padding=padding, act=paddle.activation.Linear())
    net = paddle.layer.batch_norm(input=net, act=paddle.activation.Relu())
    return paddle.layer.img_pool(input=net, pool_size=2, pool_size_y=1, stride=2, stride_y=1, pool_type=paddle.pooling.Max())

def cnn1(input,filter_size,num_channels,num_filters=64, stride=1, padding=1, act=paddle.activation.Linear()):
    return  paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels,
         num_filters=num_filters, stride=stride, padding=padding, act=act)

def printLayer(layer):
    print("depth:",layer.depth,"height:",layer.height,"width:",layer.width,"num_filters:",layer.num_filters,"size:",layer.size,"outputs:",layer.outputs)

def network():
    # 每批32张图片，将输入转为 1 * 256 * 256 CHW 
    # x = paddle.layer.data(name='x', height=1, width=2048, type=paddle.data_type.dense_vector(train_size*2048))  
    x = paddle.layer.data(name='x', height=train_size, width=2048//channels_num, type=paddle.data_type.dense_vector(train_size*2048))
    # x = paddle.layer.data(name='x', height=1, width=2048//channels_num, type=paddle.data_type.dense_vector_sequence(2048))   

    # 是否精彩分类
    a = paddle.layer.data(name='a', type=paddle.data_type.integer_value_sequence(class_dim))
    # box 分类器
    c = paddle.layer.data(name='c', type=paddle.data_type.integer_value_sequence(class_dim))
    # box 边缘修正
    b = paddle.layer.data(name='b', type=paddle.data_type.dense_vector_sequence(box_dim))

    main_nets = []
    net = cnn2(x,   3, channels_num, 64, 1, 1)    #32
    net = cnn2(net, 3, 64, 64, 1, 1)    #16
    net = cnn2(net, 3, 64, 64, 1, 1)    #8
    net = cnn2(net, 3, 64, 64, 1, 1)    #4
    net = cnn2(net, 3, 64, 64, 1, 1)    #2
    net = cnn2(net, 3, 64, 64, 1, 1)    #2
    main_nets.append(net)  
    net = cnn2(net, 3, 64, 64, 1, 1)    #1
    net = cnn2(net, 3, 64, 64, 1, 1)    #1
    main_nets.append(net)  
    # net = cnn2(net, 3, 64, 64, 1, 1)    #1
    # main_nets.append(net) 

    blocks = []
    for i  in range(len(main_nets)):
        main_net = main_nets[i]
        block_expand = paddle.layer.block_expand(input=main_net, num_channels=main_net.num_filters, 
            stride_x=1, stride_y=1, block_x=main_net.width, block_y=1)
        # block_expand_drop = paddle.layer.dropout(input=block_expand, dropout_rate=0.5)
        # blocks.append(block_expand_drop)
        blocks.append(block_expand)

    costs=[]
    net_class_gru = paddle.networks.simple_gru(input=blocks[-1], size=8, act=paddle.activation.Relu())
    net_class_fc = paddle.layer.fc(input=net_class_gru, size=class_dim, act=paddle.activation.Softmax())
    cost_class = paddle.layer.classification_cost(input=net_class_fc, label=a)

    net_box_class_fc = paddle.layer.fc(input=blocks[-1], size=class_dim, act=paddle.activation.Softmax())
    cost_box_class = paddle.layer.classification_cost(input=net_box_class_fc, label=c)

    net_box_fc = paddle.layer.fc(input=blocks, size=class_dim, act=paddle.activation.Tanh())
    cost_box = paddle.layer.square_error_cost(input=net_box_fc, label=b)

    costs.append(cost_class)
    # costs.append(cost_box_class)
    # costs.append(cost_box)
    
    parameters = paddle.parameters.create(costs)
    parameter_names = parameters.names()
    print parameter_names
    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    # return costs, parameters, adam_optimizer, net_box_class_fc, net_box_fc 
    return costs, parameters, adam_optimizer, net_box_class_fc, net_box_fc

# def read_data(v_data):
#     batch_data = np.zeros((train_size, channels_num, 2048//channels_num))  
#     w = v_data.shape[0]
#     for i in range(w):
#         _data = np.reshape(v_data[i], (1, channels_num, 2048//channels_num))
#         batch_data = np.append(batch_data[1:, :, :], _data, axis=0)
#         if i>0 and (i+1)%(train_size//4)==0:
#             fix_batch_data = np.transpose(batch_data,(1, 0, 2))
#             yield i, np.ravel(fix_batch_data)
#     if w%train_size!=0:
#         yield w-1,np.ravel(fix_batch_data)

def read_data(v_data):
    batch_data = np.zeros((train_size, 2048))  
    w = v_data.shape[0]
    for i in range(w):
        _data = np.reshape(v_data[i], (1, 2048))
        batch_data = np.append(batch_data[1:, :], _data, axis=0)
        if i>0 and (i+1)%(train_size//4)==0:
            fix_batch_data = np.reshape(batch_data,(channels_num, train_size, 2048//channels_num))
            yield i, np.ravel(fix_batch_data)
    if w%train_size!=0:
        yield w-1,np.ravel(fix_batch_data)

data_pool = []
training_data, validation_data, _ = load_data()
def readDatatoPool():
    size = len(training_data)+len(validation_data)
    c = 0
    for i in range(size):
        # if i%2==0:
        if True:
            data = random.choice(training_data)
            v_data = np.load(os.path.join(data_path,"training", "%s.pkl"%data["id"]))               
        else:
            data = random.choice(validation_data)
            v_data = np.load(os.path.join(data_path,"validation", "%s.pkl"%data["id"]))               

        # print "reading", data["id"], v_data.shape 

        for i, _data in read_data(v_data):
            fix_segments =[]
            for annotations in data["data"]:
                segment = annotations['segment']
                if segment[0]>=i or segment[1]<=i-train_size:
                    continue
                fix_segments.append([max(0, segment[0]-(i-train_size)),min(train_size-1,segment[1]-(i-train_size))])
                out_a, out_c, out_b = calc_value(fix_segments)
                data_pool.append((_data, out_a, out_c, out_b))
        while len(data_pool)>buf_size:
            print('r')
            time.sleep(1) 

# 计算 IOU,输入为 x1,x2 坐标
def calc_iou(src, dst):
    all_size = src[1]-src[0]+dst[1]-dst[0]
    full_size = max(src+dst)-min(src+dst)
    if full_size >= all_size:
        return 0
    else:
        return (all_size - full_size)/full_size

# 创建box
def get_boxs():
    boxs=[]
    for i in range(train_size):
        if i%8==0:
            for ratio in area_ratio:
                src = [max((i-block_size)*ratio, 0), min((i+block_size)*ratio, train_size-1)]
                boxs.append(src)
    return boxs

# 根据坐标返回box
def get_box_point(point):
    i = point - point%4
    ratio = area_ratio[point%4]
    return [max((i-block_size)*ratio,0), min((i+block_size)*ratio,train_size)]

# 按 block_size 格计算,前后各 block_size 格
# out_c iou 比例
# out_b 左偏移 和 右偏移
def calc_value(segments):
    out_a=[0 for _ in range(train_size)]
    out_c=[0 for _ in range(train_size)]
    out_b=[np.zeros(2) for _ in range(train_size)]

    for dst in segments:
        for i in range(int(dst[0]), int(dst[1])+1):
            out_a[i] = 1

    boxs = get_boxs()
    for i, src in enumerate(boxs):
        ious = []

        # 计算这个方框和所有的标注之间的拟合度
        for dst in segments:
            ious.append(calc_iou(src, dst))

        # 选择最大拟合度的记录下来
        max_ious = max(ious)
        max_ious_index = ious.index(max_ious)
        if max_ious>=0.5:
            out_c[i]=1
            # out_b[i][0]=(segments[max_ious_index][1]+segments[max_ious_index][0] - src[1]-src[0])/(2*train_size)
            # out_b[i][1]=(segments[max_ious_index][1]-segments[max_ious_index][0] - src[1]+src[0])/train_size 
            out_b[i][0]=(segments[max_ious_index][0]-src[0])/train_size
            out_b[i][1]=(segments[max_ious_index][1]-src[1])/train_size         
        # if max_ious > 0.9:
        #     print u"正确的:",segments[max_ious_index],u"接近的:", src, u"拟合度：", max_ious,u"偏移：", out_b[i]
    return out_a, out_c, out_b
                
def reader_get_image_and_label():
    def reader():
        t1 = threading.Thread(target=readDatatoPool, args=())
        t1.start()
        while t1.isAlive():
            while len(data_pool)==0:
                time.sleep(1)
            if len(data_pool)<buf_size//2 and random.random()>0.5:
                d, a, c, b = random.choice(data_pool)
            else:    
                d, a, c, b = data_pool.pop(random.randrange(len(data_pool)))
            # print len(d),len(a),len(c),len(b)
            # print d
            # print a
            # print c
            # print b
            yield d, a, c, b
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

# bs = get_boxs()
# src = ["-" for _ in range(128)]
# print "".join(src)
# for b in bs:
#     x = ["-" for _ in range(128)]
#     for i in range(int(b[0]),int(b[1])+1):
#         if i>=0 and i<128:
#             x[i]="+"
#     print "".join(x)

print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=2) 
paddle.init(use_gpu=True, trainer_count=1)
print("get network ...")
cost, paddle_parameters, adam_optimizer, _, _ = network()
print('set reader ...')
train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
feeding={'x':0, 'a':1, 'c':2, 'b':3}
# feeding={'x':0, 'a':1} 
# if os.path.exists(param_file):
#     (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(param_file)
#     print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
#     print("loading parameters ...")
#     paddle_parameters = paddle.parameters.Parameters.from_tar(open(param_file,"rb"))

trainer = paddle.trainer.SGD(cost=cost, parameters=paddle_parameters, update_equation=adam_optimizer)
print("start train ...")
trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=1)
