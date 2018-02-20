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
# param_file = "/home/kesci/work/param2.data"
# param_file_bak = "/home/kesci/work/param2.data.bak"
# result_json_file = "/home/kesci/work/ai2.json"

class_dim = 2 # 分类 0，背景， 1，精彩
train_size = 64 # 学习的关键帧长度
block_size = 8

buf_size = 5000
batch_size = 2048//(train_size*block_size)

model_path = os.path.join(home,"model")
cls_param_file = os.path.join(model_path,"param_cls.tar")
box_param_file = os.path.join(model_path,"param_box.tar")

result_json_file = os.path.join(model_path,"ai.json")
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
            if os.path.exists(os.path.join(training_path, "%s.pkl"%data_id)):
                training_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
        elif data['database'][data_id]['subset'] == 'validation':
            if os.path.exists(os.path.join(validation_path, "%s.pkl"%data_id)):
                validation_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
        elif data['database'][data_id]['subset'] == 'testing':
            if os.path.exists(os.path.join(testing_path, "%s.pkl"%data_id)):
                testing_data.append({'id':data_id,'data':data['database'][data_id]['annotations']})
    print('load data train %s, valid %s, test %s'%(len(training_data), len(validation_data), len(testing_data)))
    return training_data, validation_data, testing_data

training_data, validation_data, _ = load_data()

def printLayer(layer):
    print("depth:",layer.depth,"height:",layer.height,"width:",layer.width,"num_filters:",layer.num_filters,"size:",layer.size,"outputs:",layer.outputs)


def cnn(input,filter_size,num_channels,num_filters=64, stride=2, padding=1):
    return paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels, 
        num_filters=num_filters, stride=stride, padding=padding, act=paddle.activation.Relu())

def normal_network(x,drop):
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
    net = paddle.layer.img_pool(input=net, pool_size=4, pool_size_y=4, stride=1, padding=0, padding_y=0, pool_type=paddle.pooling.Avg())  
    return net

def network(drop=True):
    # 每批32张图片，将输入转为 1 * 256 * 256 CHW 
    x = paddle.layer.data(name='x', height=64, width=64, type=paddle.data_type.dense_vector_sequence(2048*block_size))   

    # 是否精彩分类
    a = paddle.layer.data(name='a', type=paddle.data_type.integer_value_sequence(class_dim))

    net = normal_network(x,drop)
    # 当前图片精彩或非精彩分类
    net_class_fc = paddle.layer.fc(input=net, size=class_dim, act=paddle.activation.Softmax())
    cost_class = paddle.layer.classification_cost(input=net_class_fc, label=a)
   
    adam_optimizer = paddle.optimizer.Adam(learning_rate=2e-3)
    return cost_class, adam_optimizer, net_class_fc

data_1 = {i:[] for i in range(10)}
data_0 = {i:[] for i in range(10)}

data_1 = {i:[] for i in range(10)}
data_0 = {i:[] for i in range(10)}

def add_zero_data_to_list(label, v_data):
    w = v_data.shape[0]
    for i in range(0, w-block_size, 2):
        if max(label[i:i+block_size]) == 0:
            yield 0, v_data[i:i+block_size]

def add_one_data_to_list(segment, label, v_data):
    w = v_data.shape[0]
    def filter(i):
        if i>=0 and i+block_size<=w: 
            if sum(label[i:i+block_size]) >= block_size*0.8:
                return 1, v_data[i:i+block_size]
        return None
    start = int(round(segment[0]))
    end = int(round(segment[1]))
    
    for i in range(start-1, start+8):
        _data = filter(i)
        if _data != None: yield _data
    for i in range(end-block_size-8, end-block_size+1):
        _data = filter(i)
        if _data != None: yield _data
            
def pre_data():
    size = len(training_data)
    datas=[]
    labels=[]
    k=0
    j=0
    for c, data in enumerate(training_data):
        v_data = np.load(os.path.join(training_path, "%s.pkl"%data["id"]))  

        w = v_data.shape[0]
        label = [0 for _ in range(w)]
        for annotations in data["data"]:
            segment = annotations['segment']
            for i in range(int(segment[0]),min(w,int(segment[1]+1))):
                label[i] = 1

        for annotations in data["data"]:
            segment = annotations['segment']
            for _l, _data in add_one_data_to_list(segment, label, v_data):
                data_1[k%10].append(_data)
                k += 1

        for _l, _data in add_zero_data_to_list(label, v_data):
            data_0[j%10].append(_data)
            j += 1

        status["progress"]="%s/%s"%(c,size)
        # print("readed %s/%s %s.pkl, size: %s/%s"%(c,size,data["id"],len(data_1[0]),len(data_0[0])))



def reader_get_image_and_label():
    def reader():
        datas=[]
        labels=[]    
        t1 = threading.Thread(target=pre_data, args=())
        t1.start()
        time.sleep(10)
        while (len(data_1[0])>1000 and len(data_0[0])>1000) or t1.isAlive(): 
            if t1.isAlive() and (len(data_1[0])<1000 or len(data_0[0])<1000):
                time.sleep(0.1)
            must_pop = False
            if random.random() > 0.5:
                labels.append(1)
                _data = data_1 
            else:
                labels.append(0)
                _data = data_0
                if not t1.isAlive(): must_pop = True

            _i = random.randint(0,9)
            _size = len(_data[_i]) 
            if _size >500 and (_size > buf_size or must_pop):
                datas.append(_data[_i].pop(0))
            else:
                datas.append(random.choice(_data[_i]))
                
            if len(datas) == train_size:
                yield datas, labels
                datas=[]
                labels=[]
    return reader


status ={}
status["starttime"]=time.time()
status["steptime"]=time.time()
status["progress"]=""
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id>0 and event.batch_id % 100 == 0:
            print "Paid %.2f,Time %.2f, Progress %s, Pass %d, Batch %d, Cost %f, %s" % (
                time.time() - status["starttime"], time.time() - status["steptime"], status["progress"], event.pass_id, 
                event.batch_id, event.cost, event.metrics) 
            status["steptime"]=time.time()
            if event.cost<10:
                cls_parameters.to_tar(open(cls_param_file, 'wb'))

def train():
    print('set reader ...')
    train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
    feeding_class={'x':0, 'a':1} 
    trainer = paddle.trainer.SGD(cost=cost, parameters=cls_parameters, update_equation=adam_optimizer)
    print("start train class ...")
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding_class, num_passes=50)
    print("paid:", time.time() - status["starttime"])
#     cls_parameters.to_tar(open(cls_param_file, 'wb'))


if __name__ == '__main__':
    print("paddle init ...")
    paddle.init(use_gpu=True, trainer_count=1)
    print("get network ...")
    cost, adam_optimizer, net_class_fc = network(True)
    cls_parameters = paddle.parameters.create(cost)
    if os.path.exists(cls_param_file):
        cls_parameters = paddle.parameters.Parameters.from_tar(open(cls_param_file,"rb"))
    train()
