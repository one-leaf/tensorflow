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
import zipfile

home = os.path.dirname(__file__)
data_path = os.path.join(home,"data")
model_path = os.path.join(home,"model")
param_file = os.path.join(model_path,"param2.tar")
result_json_file = os.path.join(model_path,"ai.json.zip")
out_dir = os.path.join(model_path, "out")

# home = "/home/kesci/work/"
# data_path = "/mnt/BROAD-datasets/video/"
# param_file = "/home/kesci/work/param2.data"
# param_file_bak = "/home/kesci/work/param2.data.bak"
# result_json_file = "/home/kesci/work/ai2.json"

if not os.path.exists(model_path): os.mkdir(model_path)
if not os.path.exists(out_dir): os.mkdir(out_dir)

class_dim = 3 # 0 不是关键 1 是关键 2 重复关键
train_size = 8 # 学习的关键帧长度

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
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(3))
   
    net = cnn(x,    8,  1, 64, 2, 2)
    net = cnn(net, 6, 64, 64, 2, 2)
    net = cnn(net, 4, 64, 64, 2, 1)
    net = cnn(net, 3, 64, 64, 2, 1)

    sliced_feature = paddle.layer.block_expand(input=net, num_channels=64, stride_x=1, stride_y=1, block_x=128, block_y=1)
    gru_forward = paddle.networks.simple_gru(input=sliced_feature, size=64, act=paddle.activation.Relu())
    gru_backward = paddle.networks.simple_gru(input=sliced_feature, size=64, act=paddle.activation.Relu(), reverse=True)
    output = paddle.layer.fc(input=[gru_forward, gru_backward], size=class_dim, act=paddle.activation.Softmax())
    
    cost = paddle.layer.classification_cost(input=output, label=y)
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    return cost, parameters, adam_optimizer, output
      
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=2) 
paddle.init(use_gpu=True, trainer_count=1)
print("get network ...")
cost, paddle_parameters, adam_optimizer, output = network()

# 预测时需要读取模型
(mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(param_file)
print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
print("loading parameters ...")
paddle_parameters = paddle.parameters.Parameters.from_tar(open(param_file,"rb"))
    

def getTestData(testFileid):
    v_data = np.load(os.path.join(data_path,"validation", "%s.pkl"%testFileid))
    data = []
    batch_data = np.zeros((2048, train_size), dtype=np.float)  
    w = v_data.shape[0]
    label = np.zeros([w], dtype=np.int)
    for i in range(w):
        _data = np.reshape(v_data[i], (2048,1))
        batch_data = np.append(batch_data[:, 1:], _data, axis=1)
        _data = np.ravel(batch_data)
        data.append((_data,))
    return data

# 转换分类到段
# 0000111111111111100000
# 0000000000001111111110
def conv_to_segment(probs):
    sort_probs = np.argsort(-probs)
    value_probs = sort_probs[:,0]
#     print(value_probs)
    w=len(value_probs)
    items = []
    minsec1 = 10
    minsec2 = 15
    avgsec = 50
    maxsec = 3600
    
# 提高高概率得分    
    # print(value_probs)
    # print(np.max(probs,axis=1))
    for i,v in enumerate(value_probs):
        if probs[i][v]>0.9:
            value_probs[i-train_size+1:i+1] = v
       
    
#     print(value_probs)
#     return items
    # 再来找正常的块
    score = 0
    start = None
    end = None
    for i, v in enumerate(value_probs):  
        # 如果 v == 1 或者往后 minsec 秒内还有>0的，都算
        _continue = False
        if start != None and w > i:
            for j in range(min(w-i, minsec1)):
                if value_probs[i+j]>0:
                    _continue = True
                    break
            if _continue == False and end - start < maxsec:
                for j in range(min(w-i, minsec2)):
                    if value_probs[i+j]>0:
                        _continue = True
                        break
                        
        if v>0 or _continue :
            if start==None:
                start = i
            end = i
            if v > 0:
                score += probs[i][v]
            else:
                score += probs[i][1]
        else:
            if start != None and end !=None:
                seg_value ={}
                seg_value["score"]=score/(end-start+1)
                seg_value["segment"]=[start, end]
                if sum(value_probs[start:end+1]) > avgsec:
                    items.append(seg_value)
                start = None
                end = None
                score = 0

    if (start != None) and (end !=None):
        seg_value ={}
        seg_value["score"]=score/(end-start+1)
        seg_value["segment"]=[start, end]
        if sum(value_probs[start:end+1]) > avgsec:
            items.append(seg_value)  
        start = None
        end = None
  
    return items

def test():
    items = []
    _, validation_data, _ = load_data("validation") 
    size = len(validation_data)
    for i, data_info in enumerate(validation_data):       
        data_id = data_info["id"]

        data = getTestData(data_id)  
        
        w = len(data)
        print("\nstart infer: %s / %s  %s size %s"%(i, size, data_id, w))
        
        all_values=[]
        batch_size = 256
        count = w // batch_size
        print("need infer count:", count)
        
        save_file = os.path.join(out_dir,data_id)
        if not os.path.exists(save_file):

            for i in range(count):
                _data = data[i*batch_size:(i+1)*batch_size]
                probs = paddle.infer(output_layer=output, parameters=paddle_parameters, input=_data)
                all_values.append(probs)
                sys.stdout.write(".")
                sys.stdout.flush()           
                
            if w%batch_size != 0:
                _data = data[count*batch_size:]
                probs = paddle.infer(output_layer=output, parameters=paddle_parameters, input=_data)
                all_values.append(probs)
                sys.stdout.write('.')
                sys.stdout.flush() 
        
            _all_values = np.row_stack(all_values)
            np.save(open(save_file,"wb"), _all_values)
        else:
            _all_values = np.load(open(save_file,"rb"))

        label = np.zeros([w], dtype=np.int)

        for annotations in data_info["data"]:
            segment = annotations['segment']
            for i in range(int(segment[0]),int(segment[1]+1)):
                label[i] += 1

        print(label[0:1000])
        print(np.argsort(-_all_values)[:,0][0:1000])
        print(np.max(_all_values,axis=1)[0:1000])

        item = conv_to_segment(_all_values)
        items.append((data_id, item))
        print(len(item))        
        del data
    return items

logger = logging.getLogger('paddle')
logger.setLevel(logging.ERROR)
np.set_printoptions(threshold=np.inf)

items = test()
result={}
result["version"]="VERSION 1.0"
result["results"]={}

for id, item in items:
    result["results"][id] = item

with zipfile.ZipFile(result_json_file,"w") as f:
    f.writestr('ai.json',json.dumps(result))
    
print("OK")
