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

class_dim = 3 # 0 不是关键 1 是关键 2 重复关键
train_size = 16 # 学习的关键帧长度

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

def resnet(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) / 6
    conv1 = conv_bn_layer(ipt, ch_in=train_size, ch_out=64, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 64, n, 2)
    res2 = layer_warp(basicblock, res1, 64, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    res4 = layer_warp(basicblock, res3, 64, n, 2)
    res5 = layer_warp(basicblock, res4, 64, n, 2)
    res6 = layer_warp(basicblock, res5, 64, n, 2)
    res7 = layer_warp(basicblock, res6, 64, n, 2)
    res8 = layer_warp(basicblock, res7, 64, n, 2)
    pool = paddle.layer.img_pool(input=res8, pool_size=8, pool_size_y=1, stride=1, padding=0, padding_y=0, pool_type=paddle.pooling.Avg())
    return pool

def network():
    # -1 ,2048*5 
    x = paddle.layer.data(name='x', width=2048, height=1, type=paddle.data_type.dense_vector(2048*train_size))
    # y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(3))
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value_sequence(class_dim))
    # y_emb = paddle.layer.embedding(input=y, size=train_size)

    layer = resnet(x, 8)
    fc = paddle.layer.fc(input=layer,size=1024)
    outputs=[]
    for i in range(train_size):
        outputs.append(paddle.layer.fc(input=fc,size=class_dim,act=paddle.activation.Softmax()))
    outputs = paddle.layer.concat(input=outputs)

    # output = paddle.layer.fc(input=outputs, size=class_dim, act=paddle.activation.Softmax())

    # output = paddle.layer.fc(input=layer,size=train_size,act=paddle.activation.Softmax())

    # sliced_feature = paddle.layer.block_expand(input=x, num_channels=train_size, stride_x=1, stride_y=1, block_x=2048, block_y=1)
    # gru_forward = paddle.networks.simple_gru(input=sliced_feature, size=64, act=paddle.activation.Relu())
    # gru_backward = paddle.networks.simple_gru(input=sliced_feature, size=64, act=paddle.activation.Relu(), reverse=True)
    # output = paddle.layer.fc(input=[gru_forward, gru_backward, layer], size=class_dim, act=paddle.activation.Softmax())
    
    cost = paddle.layer.multi_binary_label_cross_entropy_cost(input=output, label=y)
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=5e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))
    return cost, parameters, adam_optimizer, output

def reader_get_image_and_label():
    def reader():
        training_data, _, _ = load_data("training") 
        size = len(training_data)
        for i, data in enumerate(training_data):
            batch_data = np.zeros((2048, train_size))    
            v_data = np.load(os.path.join(data_path,"training", "%s.pkl"%data["id"]))               
            print("\nstart train: %s / %s %s.pkl, shape: %s"%(i, size, data["id"], v_data.shape))                
            w = v_data.shape[0]
            label = np.zeros([w], dtype=np.int)

            for annotations in data["data"]:
                segment = annotations['segment']
                for i in range(int(segment[0]),int(segment[1]+1)):
                    label[i] += 1

            for i in range(w):
                _data = np.reshape(v_data[i], (2048,1))
                batch_data = np.append(batch_data[:, 1:], _data, axis=1)
                yield np.ravel(batch_data), label[i-train_size+1:i+1]
            del v_data
    return reader

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id>0 and event.batch_id % 20 == 0:
            print("\nPass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics) )
            with open(param_file, 'wb') as f:
                paddle_parameters.to_tar(f)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
        
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=2) 
paddle.init(use_gpu=True, trainer_count=2)
print("get network ...")
cost, paddle_parameters, adam_optimizer, output = network()
print('set reader ...')
train_reader = paddle.batch(paddle.reader.shuffle(reader_get_image_and_label(), buf_size=8192), batch_size=128)
# train_reader = paddle.batch(reader_get_image_and_label(True), batch_size=64)
feeding={'x': 0, 'y': 1}
 
trainer = paddle.trainer.SGD(cost=cost, parameters=paddle_parameters, update_equation=adam_optimizer)
print("start train ...")
trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=1)
