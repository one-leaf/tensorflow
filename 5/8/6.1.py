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

class_dim = 2 # 分类 0，背景， 1，精彩， 2无效区
train_size = 64 # 学习的关键帧长度
block_size = 1024

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

def network(drop=True):
    # 全部序列的预测结果
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector_sequence(1))   

    # 是否精彩分类
    a = paddle.layer.data(name='a', type=paddle.data_type.integer_value_sequence(class_dim))

    gru_forward = paddle.networks.simple_gru(input=x, size=128, act=paddle.activation.Relu())
    gru_backward = paddle.networks.simple_gru(input=x, size=128, act=paddle.activation.Relu(), reverse=True)

    net_class_fc = paddle.layer.fc(input=[gru_forward,gru_backward], size=class_dim, act=paddle.activation.Softmax())

    cost_class = paddle.layer.classification_cost(input=net_class_fc, label=a)
   
    adam_optimizer = paddle.optimizer.Adam(learning_rate=2e-3)
    return cost_class, adam_optimizer, net_class_fc

def reader_get_image_and_label():
    def reader():
        save_file = os.path.join(out_dir,"infer.pkl")
        infers = pickle.load(open(save_file,"r"))        
        for data in training_data:               
            _x =  infers[data["id"]]
            w= len(_x)
            _x = np.array(_x)
            _x = np.reshape(_x,(w,1))
            label = [0 for _ in range(w)]

            for annotations in data["data"]:
                segment = annotations['segment']
                for i in range(int(segment[0]),min(w,int(segment[1]+1))):
                    label[i] = 1
            
            for i in range(w-block_size):
                if random.random()>0.75: 
                    yield [_x[i:i+block_size], label[i:i+block_size]]
    return reader

status ={}
status["starttime"]=time.time()
status["steptime"]=time.time()
status["progress"]=""
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id>0 and event.batch_id % 2 == 0:
            print "Paid %.2f,Time %.2f, Progress %s, Pass %d, Batch %d, Cost %f, %s" % (
                time.time() - status["starttime"], time.time() - status["steptime"], status["progress"],
                event.pass_id, event.batch_id, event.cost, event.metrics) 
            status["steptime"]=time.time()
            cls_parameters.to_tar(open(cls_param_file, 'wb'))

def train():
    print('set reader ...')
    train_reader = paddle.batch(reader_get_image_and_label(), batch_size=8)
    feeding_class={'x':0, 'a':1} 
    trainer = paddle.trainer.SGD(cost=cost, parameters=cls_parameters, update_equation=adam_optimizer)
    print("start train class ...")
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding_class, num_passes=7)
    print("paid:", time.time() - status["starttime"])
#     cls_parameters.to_tar(open(cls_param_file, 'wb'))


if __name__ == '__main__':
    print("paddle init ...")
    paddle.init(use_gpu=True, trainer_count=1)
    print("get network ...")
    cost, adam_optimizer, net_class_fc = network(True)
    cls_parameters = paddle.parameters.create(cost)
    train()
