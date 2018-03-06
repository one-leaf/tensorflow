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
import pickle
import commands, re  
import threading
from collections import deque

home = "/home/kesci/work/"
data_path = "/mnt/BROAD-datasets/video/"

class_dim = 2 # 分类 0，空白  1 过程，2，开始, 3，结束， 
# class_dim = 3 # 分类 0，空白  2，开始, 3，结束， 
# class_dim = 2  # 分类 0，空白  1 过程
train_size = 32 # 学习的关键帧长度
block_size = 2  # 坐标在右边，需要为偶数
mark_length = 8 # 标记
learning_rate = 1e-3

buf_size = 10000
batch_size = 2048//(train_size*block_size)
max_train_time = 3600*3

model_path = os.path.join(home,"model")
cls_param_file = os.path.join(model_path,"param_cls_%s.tar"%class_dim)
status_file = os.path.join(model_path,"status_%s.json"%class_dim)

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

training_data, validation_data, testing_data = load_data()

def printLayer(layer):
    print("depth:",layer.depth,"height:",layer.height,"width:",layer.width,"num_filters:",layer.num_filters,"size:",layer.size,"outputs:",layer.outputs)


def cnn(input,filter_size,num_channels,num_filters=64, stride=2, padding=1):
    return paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels, 
        num_filters=num_filters, stride=stride, padding=padding, act=paddle.activation.Relu())

def pool(input, pool_size=2):
    return paddle.layer.img_pool(input=input, pool_size=pool_size, pool_size_y=pool_size, 
                                 stride=2, padding=0, padding_y=0, pool_type=paddle.pooling.Avg())  
   
def lstm(input, fc_size):
    fc_para_attr = paddle.attr.Param(learning_rate=learning_rate)
    lstm_para_attr = paddle.attr.Param(initial_std=0., learning_rate=1.)
    para_attr = [fc_para_attr, lstm_para_attr]
    bias_attr = paddle.attr.Param(initial_std=0., l2_rate=0.)
    relu = paddle.activation.Relu()
    linear = paddle.activation.Linear()
    
    fc1 = paddle.layer.fc(input=input, size=fc_size, act=linear, bias_attr=bias_attr, 
            layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=drop_rate))
    lstm1 = paddle.layer.lstmemory(input=fc1, act=relu, bias_attr=bias_attr)
    inputs = [fc1, lstm1]
    for i in range(2, 4):
        fc = paddle.layer.fc(input=inputs, size=fc_size, act=linear, param_attr=para_attr, 
                bias_attr=bias_attr, layer_attr=paddle.attr.ExtraLayerAttribute(drop_rate=drop_rate))
        lstm = paddle.layer.lstmemory(input=fc, reverse=(i % 2) == 0, act=relu, bias_attr=bias_attr)
        inputs = [fc, lstm]        
    return [gru_forward, gru_backward]    
    
def normal_network(x):
    net = cnn(x,     5, 1, 64, 2, 2) 
    net = pool(net)
    net = cnn(net,  5, 64, 128, 2, 2)
    net = cnn(net,  3, 128, 256, 2, 1)
    net = cnn(net,  2, 256, 512, 2, 0)
    net = cnn(net,  1, 512, 256, 1, 0)
#     net = paddle.layer.batch_norm(input=net, act=paddle.activation.Linear())
    net = cnn(net,  1, 256, 128, 1, 0)
#     net = paddle.layer.batch_norm(input=net, act=paddle.activation.Linear())
    return net

def network():
    x = paddle.layer.data(name='x', height=32, width=32, type=paddle.data_type.dense_vector_sequence(2048*block_size))   

    a = paddle.layer.data(name='a', type=paddle.data_type.integer_value_sequence(class_dim))

    net = cnn(x,     8, 2048*block_size//32//32, 64, 2, 0)     
    net = cnn(net,   4, 64, 64, 2, 0)     
    gru_forward = paddle.networks.simple_gru(input=net, size=512, act=paddle.activation.Relu())
    gru_backward = paddle.networks.simple_gru(input=net, size=512, act=paddle.activation.Relu(), reverse=True)
    output = paddle.layer.concat([gru_forward,gru_backward])
    net = normal_network(output)
        
    net_class_fc = paddle.layer.fc(input=net, size=class_dim, act=paddle.activation.Softmax())
    cost_class = paddle.layer.classification_cost(input=net_class_fc, label=a)

    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=learning_rate,
        learning_rate_decay_a=0.5,
        learning_rate_decay_b=0.75,
        learning_rate_schedule="poly",
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))
    return cost_class, adam_optimizer, net_class_fc

data_0 = deque(maxlen=buf_size)
data_1 = deque(maxlen=buf_size)
data_2 = deque(maxlen=buf_size*8)

def get_training_data():
    data_ids = sorted(status["training_data_values"].items(), key=lambda x:x[1])

    # 取误差最低的50%数据学习
    get_rate = len(data_ids)/2 
    
    data_names = []
    count = 0
    for _data_id,_data_values in data_ids:
        if _data_values!=0 and count < get_rate:
            data_names.append(_data_id)
            count += 1
        elif _data_values==0:
            data_names.append(_data_id)
        
    data_id = random.choice(data_names)
    if status["training_data_values"][data_id]==0:
        status["training_data_values"][data_id]==1
    for data in training_data:
        if data["id"]==data_id:
            return data

def add_data_to_list(label, v_data, random_rate): 
    rate = 1.0*min([len(data_0),len(data_1),len(data_2)])/buf_size
    if rate < 0.8 and rate > 0.2:
        count = 2
    else:
        count = 1
    for j in range(count):
        w = v_data.shape[0]
        need_remove_index=[]
        for i in range(0,w):
            if (label[i]==0 or label[i]==1) and random.random()>random_rate: 
                need_remove_index.append(i)
        _labels= np.delete(label, need_remove_index)
        _v_datas= np.delete(v_data, need_remove_index, axis=0)
        start = random.randint(0,train_size)        

        w = _v_datas.shape[0]
        _data=[]
        _label=[]
        for i in range(start, w-block_size):
            one_data = _v_datas[i:i+block_size]
            assert one_data.shape == (block_size,2048)
            _data.append(one_data)
            if  _labels[i]>0 :
                _label.append(1)
            else:
                _label.append(0)
            if len(_data)==train_size:
                yield _data, _label
                _data = []
                _label = []

def pre_data():
    size = len(training_data)
    datas=[]
    labels=[]
    while True:
        t_data = get_training_data()
        v_data = np.load(os.path.join(training_path, "%s.pkl"%t_data["id"]))  
        w = v_data.shape[0]
        label = np.zeros(w, dtype=np.int8)
        for annotations in t_data["data"]:
            segment = annotations['segment']
            start = int(round(segment[0]))
            end = int(round(segment[1]))
            for i in range(start, end):
                if i<0 or i>=w: continue                  
                if i>=start and i<start+mark_length: 
                    label[i] = 2
                elif i>end-mark_length and i<=end:
                    label[i] = 3
                elif label[i] == 0:
                    label[i] = 1
        
        if status["progress"]>size:
            rate=0.9
        else:
            rate=0.95
        for _data, _label in add_data_to_list(label, v_data, rate):
            
            #只学干净的数据
            _temp_str="".join(map(str,_label))                
            if _temp_str.find("12")>0: continue
            if _temp_str.find("23")>0: continue
            if _temp_str.find("31")>0: continue
            if _temp_str.find("32")>0: continue
            if re.match(r'10{1,5}1',_temp_str): continue
            if re.match(r'20{1,5}3',_temp_str): continue
            if re.match(r'30{1,5}2',_temp_str): continue
            
            _rate = 1.0*sum(_label)/train_size
            if _rate<=0.05:
                data = data_0
            elif _rate>=0.95:
                data = data_1
            else:
                data = data_2
            data.append((t_data["id"], _data, _label))

        status["progress"] +=1
       
        # 为了公平，只学习3小时
        if status["usedtime"] > max_train_time: break
#         print("readed %s/%s %s.pkl, size: %s/%s"%(c,size,t_data["id"],len(data_0[0]),len(data_1[0])))

def reader_get_image_and_label():
    def reader():
        t1 = threading.Thread(target=pre_data, args=())
        t1.start()
        while len(data_0)<1000 or len(data_1)<1000 or len(data_2)<1000:
            print("cacheing", len(data_0), len(data_1), len(data_2))
            time.sleep(5)
            
        status["steptime"]=time.time()    
        while t1.isAlive(): 
            k=random.random()
            if k<0.2:
                data = data_0
            elif k>0.7:
                data = data_1
            else:
                data = data_2            
            _data_id, _data, _lable = random.choice(data)
            status["curr_batch_ids"].append(_data_id)
            yield _data, _lable
    return reader

status ={}
status["starttime"]=time.time()
status["steptime"]=time.time()
status["usedtime"]=0
status["progress"]=0
status["cost"]=0
status["costcount"]=0
status["size"]=3*len(training_data)
status["curr_batch_ids"]=[]
status["training_data_values"]={}
# 数据中存在脏数据，所以需要清理, 前面为cost合计，后面是训练次数
for data in training_data:
    status["training_data_values"][data['id']]=0
    
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        #记下每次训练的打分
        if event.batch_id>0:
            status["cost"] += event.cost
            status["costcount"] += 1
            cost_avg = 1.0*status["cost"]/status["costcount"] 
            while len(status["curr_batch_ids"])>0:
                data_id = status["curr_batch_ids"].pop(0)
                status["training_data_values"][data_id] += event.cost - cost_avg
        
        if event.batch_id>0 and event.batch_id % (batch_size*block_size) == 0:
            print "Paid %.2f,Time %.2f, %.2f, Batch %d, Cost %.2f/%.2f, %s" % (
                status["usedtime"], time.time() - status["steptime"], 
                1.0*status["progress"]/status["size"],
                event.batch_id, event.cost, cost_avg, event.metrics) 
            status["usedtime"] = status["usedtime"]+time.time() - status["steptime"]
            status["steptime"] = time.time()

        if event.batch_id>0 and event.batch_id % 500 == 0:
            data_ids = sorted(status["training_data_values"].items(), key=lambda x:x[1])
            for data_item in data_ids[:3]:
                print(data_item[0],data_item[1])
            for data_item in data_ids[-3:]:
                print(data_item[0],data_item[1])
            for key in status["training_data_values"]:
                status["training_data_values"][key] *= 0.5
            print("data_0: %d data_1: %d data_2: %d"%(len(data_0),len(data_1),len(data_2)))
            cls_parameters.to_tar(open(cls_param_file, 'wb'))
            json.dump(status, open(status_file,'w'))
                             
def train():
    print('set reader ...')

#     shuffle_reader = paddle.reader.shuffle(reader_get_image_and_label(), buf_size=200)
    train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
    feeding_class={'x':0, 'a':1} 
    trainer = paddle.trainer.SGD(cost=cost, parameters=cls_parameters, update_equation=adam_optimizer)
    print("start train class ...")
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding_class, num_passes=1)
    print("paid:", time.time() - status["starttime"])
    cls_parameters.to_tar(open(cls_param_file, 'wb'))

def infer():
    inferer = paddle.inference.Inference(output_layer=net_class_fc, parameters=cls_parameters)
    save_file = os.path.join(out_dir,"test.pkl")
    infers={}
    for data in testing_data:
        filename = "%s.pkl"%data["id"]
        v_data = np.load(os.path.join(testing_path, filename))
        w = v_data.shape[0]
        values = []
        datas=[]
        for i in range(w-block_size):
            datas.append(v_data[i:i+block_size])
            if len(datas)==train_size:
                probs = inferer.infer(input=[(datas,)])
                for p in probs:
                    values.append(p)
                datas=[]
        if len(datas)>0:
            probs = inferer.infer(input=[(datas,)])
            for p in probs:
                values.append(p)

        infers[data["id"]]=values
        print("infered %s"%filename)
    # print(infers)
    pickle.dump(infers,open(save_file,"wb"))
    
def infer_validation():
    inferer = paddle.inference.Inference(output_layer=net_class_fc, parameters=cls_parameters)
    infers={}
    save_file = os.path.join(out_dir,"validation.pkl")
    for data in validation_data:
        filename = "%s.pkl"%data["id"]
        v_data = np.load(os.path.join(validation_path, filename))
        w = v_data.shape[0]
        values = []
        datas=[]
        for i in range(w-block_size):
            datas.append(v_data[i:i+block_size])
            if len(datas)==train_size:
                probs = inferer.infer(input=[(datas,)])
                for p in probs:
                    values.append(p)
                datas=[]
        if len(datas)>0:
            probs = inferer.infer(input=[(datas,)])
            for p in probs:
                values.append(p)

        infers[data["id"]]=values
        print("infered %s"%filename)
    pickle.dump(infers,open(save_file,"wb"))
    # print(infers)
    
if __name__ == '__main__':
    print("paddle init ...")
    paddle.init(use_gpu=True, trainer_count=1)
    cost, adam_optimizer, net_class_fc = network()

    if os.path.exists(cls_param_file):
        print("load %s, continue train ..."%cls_param_file)
        cls_parameters = paddle.parameters.Parameters.from_tar(open(cls_param_file,"rb"))
        status = json.load(open(status_file,'r'))
        status["curr_batch_ids"]=[]
        print("has train %s"% status["usedtime"])
        if status["usedtime"] > max_train_time:
            infer_validation()
#             infer()
        else:
            train()
    else:
        cls_parameters = paddle.parameters.create(cost)
        for name in cls_parameters.names():
            print(name, cls_parameters.get_shape(name))
        train()
    print("OK")