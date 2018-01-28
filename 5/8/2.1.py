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
param_file_bak = os.path.join(model_path,"param2.tar,bak")
result_json_file = os.path.join(model_path,"ai2.json.zip")
out_dir = os.path.join(model_path, "out")

# home = "/home/kesci/work/"
# data_path = "/mnt/BROAD-datasets/video/"
# param_file = "/home/kesci/work/param2.data"
# param_file_bak = "/home/kesci/work/param2.data.bak"
# result_json_file = "/home/kesci/work/ai2.json"

if not os.path.exists(model_path): os.mkdir(model_path)
if not os.path.exists(out_dir): os.mkdir(out_dir)

class_dim = 3 # 0 不是关键 1 是关键 2 重复关键
train_size = 64 # 学习的关键帧长度

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

def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(), ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=(filter_size,1),
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=(padding,0),
        act=paddle.activation.Linear(),
        bias_attr=False)
    return paddle.layer.batch_norm(input=tmp, act=active_type)

def shortcut(ipt, n_in, n_out, stride):
    if n_in != n_out:
        return conv_bn_layer(ipt, n_out, 1, stride, 0, paddle.activation.Linear())
    else:
        return ipt

def basicblock(ipt, ch_out, stride):
    ch_in = ch_out * 2
    tmp = conv_bn_layer(ipt, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, paddle.activation.Linear())
    short = shortcut(ipt, ch_in, ch_out, stride)
    return paddle.layer.addto(input=[tmp, short], act=paddle.activation.Relu())

def layer_warp(block_func, ipt, features, count, stride):
    tmp = block_func(ipt, features, stride)
    for i in range(1, count):
        tmp = block_func(tmp, features, 1)
    return tmp

def resnet_cifar10(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(ipt, ch_in=train_size, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 2)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(input=res3, pool_size=4, pool_size_y=1, stride=2, padding=1, padding_y=0, pool_type=paddle.pooling.Avg())
    return pool

def network():
    # -1 ,2048*5 
    x = paddle.layer.data(name='x', width=2048, height=1, type=paddle.data_type.dense_vector(2048*train_size))
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(3))

    layer = resnet_cifar10(x,20)
    # output = paddle.layer.fc(input=layer,size=class_dim,act=paddle.activation.Softmax())

    sliced_feature = paddle.layer.block_expand(input=layer, num_channels=64, stride_x=1, stride_y=1, block_x=128, block_y=1)

    gru_forward = paddle.networks.simple_gru(input=sliced_feature, size=128, act=paddle.activation.Relu())
    gru_backward = paddle.networks.simple_gru(input=sliced_feature, size=128, act=paddle.activation.Relu(), reverse=True)

    output = paddle.layer.fc(input=[gru_forward,gru_backward], size=class_dim, act=paddle.activation.Softmax())
    
    cost = paddle.layer.classification_cost(input=output, label=y)
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=2e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))
    return cost, parameters, adam_optimizer, output
      
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=2) 
paddle.init(use_gpu=True, trainer_count=1)
print("get network ...")
cost, paddle_parameters, adam_optimizer, output = network()

# 预测时需要读取模型
if os.path.exists(param_file):
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
        _data1 = np.ravel(batch_data)
        data.append((_data1,))
    return data

# 转换分类到段
# 0000111111111111100000
# 0000000000001111111110
def conv_to_segment(probs):
    sort_probs = np.argsort(-probs)
    value_probs = [v[0] for v in sort_probs]
#     print(value_probs)
    w=len(value_probs)
    items = []
    minsec1 = 10
    minsec2 = 20
    avgsec = 30
    maxsec = 3600
    
    score = 0
    start = None
    end = None
    #先找重合的块：
    for i, v in enumerate(value_probs):  
        # 如果 v == 2 或者往后 minsec 秒内还有>0的，都算
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
                        
        if v==2 or _continue :
            if start==None:
                start = i
            end = i
            if v > 0:
                score += probs[i][v]
            else:
                score += probs[i][1]
        else:
            if start!=None:
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

    for seg_value in items:
        start,end = seg_value["segment"]
        for i in range(start,end):
            value_probs[i] -= 1 

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
        json_file = os.path.join(out_dir,data_id)
        if os.path.exists(json_file):
            continue
        data = getTestData(data_id)  
        
        w = len(data)
        print("\nstart infer: %s / %s  %s size %s"%(i, size, data_id, w))
        
        all_values=[]
        batch_size = 128
        count = w // batch_size
        print("need infer count:", count)
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
            sys.stdout.write('E')
            sys.stdout.flush() 
       
        _all_values = np.row_stack(all_values)
        item = conv_to_segment(_all_values)
        items.append((data_id, item))
        
        del data
    return items

logger = logging.getLogger('paddle')
logger.setLevel(logging.ERROR)

items = test()
result={}
result["version"]="VERSION 1.0"
result["results"]={}

for id, item in items:
    result["results"][id] = item

with zipfile.ZipFile(result_json_file,"w") as f:
    json.dump(result,f)
    
print("OK")
