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
box_dim = 2 # 偏移，左，右
train_size = 256 # 学习的关键帧长度
block_size = 8

buf_size = 4096
batch_size = 2048//(train_size*block_size)
box_size = train_size//4
area_ratio = (1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0.125)

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
is_trin_box=False

static_bias_attr = paddle.attr.ParamAttr(is_static=True)

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

def printLayer(layer):
    print("depth:",layer.depth,"height:",layer.height,"width:",layer.width,"num_filters:",layer.num_filters,"size:",layer.size,"outputs:",layer.outputs)

def network(drop=True):
    # 每批32张图片，将输入转为 1 * 256 * 256 CHW 
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector_sequence(1))   

    # box 分类器
    c = paddle.layer.data(name='c', type=paddle.data_type.integer_value_sequence(class_dim))
    # box 边缘修正
    b = paddle.layer.data(name='b', type=paddle.data_type.dense_vector_sequence(box_dim))

    # BOX位置是否是背景和还是有效区域分类
    net_box_class_gru = paddle.networks.simple_gru(input=x, size=128, act=paddle.activation.Relu())
    net_box_class_gru_r = paddle.networks.simple_gru(input=x, size=128, act=paddle.activation.Relu(), reverse=True)
    net_box_class_fc = paddle.layer.fc(input=[net_box_class_gru,net_box_class_gru_r], size=class_dim, act=paddle.activation.Softmax())

    # BOX的偏移量回归预测
    net_box_gru = paddle.networks.simple_gru(input=x, size=train_size, act=paddle.activation.Tanh())
    net_box_gru_r = paddle.networks.simple_gru(input=x, size=train_size, act=paddle.activation.Tanh(), reverse=True)
    net_box_fc = paddle.layer.fc(input=[net_box_gru,net_box_gru_r], size=box_dim, act=paddle.activation.Tanh())

    cost_box_class = paddle.layer.classification_cost(input=net_box_class_fc, label=c)
    cost_box = paddle.layer.square_error_cost(input=net_box_fc, label=b)

    costs=[]
    costs.append(cost_box_class)
    costs.append(cost_box)
    
    adam_optimizer = paddle.optimizer.Adam(learning_rate=2e-3,
        learning_rate_schedule="pass_manual", learning_rate_args="1:1.0,2:0.9,3:0.8,4:0.7,5:0.6,6:0.5",)
    return costs, adam_optimizer, net_box_class_fc, net_box_fc

# 读取BOX数据，连续数据
def read_data_box(v_data):
    batch_data = [np.zeros(2048*block_size) for _ in range(train_size)]   
    w = v_data.shape[0]
    for i in range(block_size, w):
        _data = np.stack([v_data[j] for j in range(i-block_size,i)])
        batch_data.append(_data)
        batch_data.pop(0)
        if i>=train_size and random.random()>1./16:
            yield i, batch_data
    if w%train_size!=0:
        yield w-1, batch_data

data_pool_0 = []    #负样本
data_pool_1 = []    #正样本
training_data, validation_data, _ = load_data()
def readDatatoPool():
    size = len(training_data)+len(validation_data)
    c = 0
    for i in range(size):
        if i%2==0:
            data = random.choice(training_data)
            v_data = np.load(os.path.join(out_dir, "%s.pkl"%data["id"]))               
        else:
            data = random.choice(validation_data)
            v_data = np.load(os.path.join(out_dir, "%s.pkl"%data["id"]))               

        print "reading", data["id"], v_data.shape , len(data_pool_0), len(data_pool_1)
        for i, _data in read_data_box(v_data):
            fix_segments =[]
            for annotations in data["data"]:
                segment = annotations['segment']
                if segment[0]>=i or segment[1]<=i-train_size:
                    continue
                fix_segments.append([max(0, segment[0]-(i-train_size)),min(train_size-1,segment[1]-(i-train_size))])
                out_c, out_b = calc_value(fix_segments)
                if random.random() > 0.5:                   
                    data_pool_0.append((_data, out_c, out_b))
                else:
                    data_pool_1.append((_data, out_c, out_b))

        while len(data_pool_1)>buf_size:
            # print("r", len(data_pool_0), len(data_pool_1))
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
                off = box_size * ratio
                src = [max(i-off, 0), min(i+off, train_size-1)]
                boxs.append(src)
    return boxs

# 根据坐标返回box
def get_box_point(point):
    i = point - point%4
    ratio = area_ratio[point%4]
    off = box_size * ratio
    return [max(i-off,0), min(i+off,train_size-1)]

# 按 box_size 格计算,前后各 box_size 格
# out_c iou 比例
# out_b 左偏移 和 右偏移
def calc_value(segments):
    out_c=[0 for _ in range(train_size)]
    out_b=[np.zeros(2) for _ in range(train_size)]

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
    return out_c, out_b
                
def reader_get_image_and_label():
    def reader():
        t1 = threading.Thread(target=readDatatoPool, args=())
        t1.start()
        while t1.isAlive():            
            if is_trin_box:
                while len(data_pool_1)==0:
                    time.sleep(1)
                if random.random()>0.5 and len(data_pool_0)>0:
                    data_pool = data_pool_0
                    v = 1.0*len(data_pool_0)/len(data_pool_1)
                else:
                    data_pool = data_pool_1
                    if len(data_pool_0)!=0:
                        v = 1.0*len(data_pool_1)/len(data_pool_0)
                    else: 
                        v = 1.0
                if random.random()>v:
                    d, c, b = random.choice(data_pool)
                else:    
                    d, c, b = data_pool.pop(random.randrange(len(data_pool)))
                yield d, c, b
            else:
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
            if is_trin_box:
                box_parameters.to_tar(open(box_param_file, 'wb'))
            else:
                cls_parameters.to_tar(open(cls_param_file, 'wb'))

# for i in range(train_size):
#     print(i,get_box_point(i)) 

if __name__ == '__main__':
    print("paddle init ...")
    paddle.init(use_gpu=True, trainer_count=1)
    print("get network ...")
    costs, adam_optimizer, net_box_class_fc, net_box_fc = network()


    if os.path.exists(box_param_file):
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(box_param_file)
        print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
        print("loading box parameters %s ..."%box_param_file)
        box_parameters = paddle.parameters.Parameters.from_tar(open(box_param_file,"rb"))
    else:
        box_parameters = paddle.parameters.create(costs)
    

    print('set reader ...')
    train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
    feeding_box={'x':0, 'c':1, 'b':2}

    is_trin_box = True
    trainer = paddle.trainer.SGD(cost=costs, parameters=box_parameters, update_equation=adam_optimizer)
    print("start train box ...")
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding_box, num_passes=1)
    print("paid:", time.time() - status["starttime"])    
