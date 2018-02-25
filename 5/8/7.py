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

home = os.path.dirname(__file__)
data_path = os.path.join(home,"data")
# home = "/home/kesci/work/"
# data_path = "/mnt/BROAD-datasets/video/"
# param_file = "/home/kesci/work/param2.data"
# param_file_bak = "/home/kesci/work/param2.data.bak"
# result_json_file = "/home/kesci/work/ai2.json"

class_dim = 4 # 分类 0，空白  1 开始， 2，过程， 3，结束
train_size = 128 # 学习的关键帧长度
block_size = 4

buf_size = 5000
batch_size = 2048*2//(train_size*block_size)

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

training_data, validation_data, _ = load_data()

def printLayer(layer):
    print("depth:",layer.depth,"height:",layer.height,"width:",layer.width,"num_filters:",layer.num_filters,"size:",layer.size,"outputs:",layer.outputs)


def cnn(input,filter_size,num_channels,num_filters=64, stride=2, padding=1):
    return paddle.layer.img_conv(input=input, filter_size=filter_size, num_channels=num_channels, 
        num_filters=num_filters, stride=stride, padding=padding, act=paddle.activation.Relu())

def normal_network(x,drop):
    net = cnn(x,    8, 2048*block_size//64//64, 64, 2, 3)
    if drop:
        net = paddle.layer.dropout(input=net, dropout_rate=0.1)
    net = cnn(net,  6, 64, 64, 2, 2)
    if drop:
        net = paddle.layer.dropout(input=net, dropout_rate=0.1)    
    net = cnn(net,  4, 64, 64, 2, 1)
    if drop:
        net = paddle.layer.dropout(input=net, dropout_rate=0.1)            
    net = cnn(net,  3, 64, 64, 2, 1) 
    net = paddle.layer.img_pool(input=net, pool_size=4, pool_size_y=4, stride=1, padding=0, padding_y=0, pool_type=paddle.pooling.Avg())  
    return net

def network(drop=True):
    x = paddle.layer.data(name='x', height=64, width=64, type=paddle.data_type.dense_vector_sequence(2048*block_size))   

    a = paddle.layer.data(name='a', type=paddle.data_type.integer_value_sequence(class_dim))

    net = normal_network(x, drop)
    # 当前图片精彩或非精彩分类
    gru_forward = paddle.networks.simple_gru(input=net, size=8, act=paddle.activation.Relu())
    gru_backward = paddle.networks.simple_gru(input=net, size=8, act=paddle.activation.Relu(), reverse=True)

    net_class_fc = paddle.layer.fc(input=[gru_forward, gru_backward], size=class_dim, act=paddle.activation.Softmax())
    cost_class = paddle.layer.classification_cost(input=net_class_fc, label=a)
   
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))
    return cost_class, adam_optimizer, net_class_fc

data_0 = {i:[] for i in range(10)}
data_1 = {i:[] for i in range(10)}
    
def add_data_to_list(label, v_data):
    w = v_data.shape[0]
    _data = []
    for i in range(random.randint(0, block_size), w-block_size):
        _data.append(v_data[i:i+block_size])
        if len(_data)==train_size:
            yield _data, label[i+1-train_size:i+1]
            _data = []

def pre_data():
    size = len(training_data)
    datas=[]
    labels=[]
    k=0
    j=0
    l=0
    for c, t_data in enumerate(training_data):
        # 由于网页训练中途会莫名中断，只能随机选择
        t_data = random.choice(training_data)
        v_data = np.load(os.path.join(training_path, "%s.pkl"%t_data["id"]))  
        w = v_data.shape[0]
        label = [0 for _ in range(w)]
        for annotations in t_data["data"]:
            segment = annotations['segment']
            start = int(round(segment[0]))
            end = int(round(segment[1]))
            for i in range(start-block_size, end+1):
                if i <0 or i>=w: continue
                if i+block_size>start and i<=start: 
                    label[i] = 1
                elif i+block_size>end and i<=end:
                    label[i] = 3
                elif label[i] == 0:
                    label[i] = 2
                    
        for _data, _label in add_data_to_list(label, v_data):
            if max(_label)==0:
                data_0[j%10].append((_data,_label))
                j += 1
            else:
                data_1[k%10].append((_data,_label))
                k += 1

        status["progress"]="%s/%s"%(c,size)

#         print("readed %s/%s %s.pkl, size: %s/%s"%(c,size,data["id"],len(data_1),len(data_0)))

def reader_get_image_and_label():
    def reader():
        t1 = threading.Thread(target=pre_data, args=())
        t1.start()
        time.sleep(10)
        while t1.isAlive(): 
            if t1.isAlive() and (len(data_0[0])<1000 or len(data_1[0])<1000):
                time.sleep(0.1)

            _i = random.randint(0,9)
            
            if random.random()>0.5:
                data = data_0
            else:
                data = data_1
            _size = len(data[_i]) 
            if _size > buf_size:
                yield data[_i].pop(0)
            else:
                yield random.choice(data[_i])
    return reader

status ={}
status["starttime"]=time.time()
status["steptime"]=time.time()
status["progress"]=""
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id>0 and event.batch_id % batch_size == 0:
            print "Paid %.2f,Time %.2f, %s, Pass %d, Batch %d, Cost %.2f, %s" % (
                time.time() - status["starttime"], time.time() - status["steptime"], status["progress"],
                event.pass_id, event.batch_id, event.cost, event.metrics) 
            status["steptime"]=time.time()
            cls_parameters.to_tar(open(cls_param_file, 'wb'))
            json.dump(status, open(status_file,'w'))
                
            # 莫名其妙数据丢失，每3000后重新跑
#             if event.batch_id==3000: exit()
            # 为了公平，只学习3小时
            if time.time() - status["starttime"] > 3600*3:
                exit()
                
def train():
    print('set reader ...')
    train_reader = paddle.batch(reader_get_image_and_label(), batch_size=batch_size)
    feeding_class={'x':0, 'a':1} 
    trainer = paddle.trainer.SGD(cost=cost, parameters=cls_parameters, update_equation=adam_optimizer)
    print("start train class ...")
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding_class, num_passes=10)
    print("paid:", time.time() - status["starttime"])
#     cls_parameters.to_tar(open(cls_param_file, 'wb'))

def infer():
    inferer = paddle.inference.Inference(output_layer=net_class_fc, parameters=cls_parameters)
    save_file = os.path.join(out_dir,"infer.pkl")
    infers={}
    for data in training_data:
        filename = "%s.pkl"%data["id"]
        v_data = np.load(os.path.join(training_path, filename))
        w = v_data.shape[0]
        values = []
        datas=[]
        for i in range(w-block_size):
            datas.append(v_data[i:i+block_size])
            if len(datas)==train_size*batch_size:
                probs = inferer.infer(input=[(datas,)])
                for p in probs:
                    values.append(p[1])
                datas=[]
        if len(datas)>0:
            probs = inferer.infer(input=[(datas,)])
            for p in probs:
                values.append(p[1])

        infers[data["id"]]=values
        print("infered %s"%filename)
    # print(infers)
    pickle.dump(infers,open(save_file,"w"))

if __name__ == '__main__':
    print("paddle init ...")
    paddle.init(use_gpu=True, trainer_count=1)
    print("get network ...")
    cost, adam_optimizer, net_class_fc = network(True)
    if os.path.exists(cls_param_file):
        print("load %s, continue train ..."%cls_param_file)
        cls_parameters = paddle.parameters.Parameters.from_tar(open(cls_param_file,"rb"))
        status = json.load(open(status_file,'r'))
    else:
        cls_parameters = paddle.parameters.create(cost)
    train()
    #infer()