# coding=utf-8
#!/sbin/python

import numpy as np
import paddle.v2 as paddle
import json
import os
import random
import sys

curr_dir = os.path.dirname(__file__)

data_path = os.path.join(curr_dir,"data")
param_file = "/home/kesci/work/param.data"

class_dim = 3 # 0 不是关键 1 是关键 2 重复关键

TEST = True

def load_data():
    data = json.loads(open(os.path.join(data_path,"meta.json")).read())
    training_data = []
    validation_data = []
    testing_data = []
    for data_id in data['database']:
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

def load_data_test():
    data = json.loads(open(os.path.join(data_path,"meta.json")).read())
    training_data = []
    validation_data = []
    testing_data = []
    _training_data = json.loads(open(os.path.join(data_path,"training","data.json")).read())
    _validation_data = json.loads(open(os.path.join(data_path,"validation","data.json")).read())
    _testing_data = json.loads(open(os.path.join(data_path,"testing","data.json")).read())
    for data_id in data['database']:
        f = "%s.pkl"%data_id
        if data['database'][data_id]['subset'] == 'training':
            if f in _training_data:
                training_data.append({'id':data_id,'data':data['database'][data_id]['annotations'],"shape":_training_data[f]})
                break
        elif data['database'][data_id]['subset'] == 'validation':
            if f in _validation_data:
                validation_data.append({'id':data_id,'data':data['database'][data_id]['annotations'],"shape":_validation_data[f]})
        elif data['database'][data_id]['subset'] == 'testing':
            if f in _testing_data:
                testing_data.append({'id':data_id,'data':data['database'][data_id]['annotations'],"shape":_testing_data[f]})
    print('load data train %s, valid %s, test %s'%(len(training_data), len(validation_data), len(testing_data)))
    return training_data, validation_data, testing_data


def conv_bn_layer(input, ch_out, filter_size, stride, padding, active_type=paddle.activation.Relu(), ch_in=None):
    tmp = paddle.layer.img_conv(
        input=input,
        filter_size=filter_size,
        num_channels=ch_in,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
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
    conv1 = conv_bn_layer(ipt, ch_in=3, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 64, n, 2)
    pool = paddle.layer.img_pool(input=res3, pool_size=7, stride=1, pool_type=paddle.pooling.Avg())
    return pool

def simple_cnn(ipt):
    layer = paddle.layer.img_conv(
        input=ipt, filter_size=5, num_filters=16, num_channels=3, 
        padding=2, stride=1, act=paddle.activation.Relu())
    layer = paddle.layer.dropout(input=layer, dropout_rate=0.2)

    layer = paddle.layer.img_conv(
        input=layer, filter_size=5, num_filters=32, num_channels=16, 
        padding=2, stride=1, act=paddle.activation.Relu())
    layer = paddle.layer.dropout(input=layer, dropout_rate=0.2)

    layer = paddle.layer.img_pool(input=layer, pool_size=2, stride=2, pool_type=paddle.pooling.Max())

    layer = paddle.layer.img_conv(
        input=layer, filter_size=3, num_filters=64, num_channels=32, 
        padding=1, stride=1, act=paddle.activation.Relu())
    layer = paddle.layer.dropout(input=layer, dropout_rate=0.2)
    
    layer = paddle.layer.img_pool(input=layer, pool_size=2, stride=2, pool_type=paddle.pooling.Max())

    layer = paddle.layer.img_conv(
        input=layer, filter_size=3, num_filters=64, num_channels=64, 
        padding=1, stride=1, act=paddle.activation.Relu())
    layer = paddle.layer.dropout(input=layer, dropout_rate=0.2)

    layer = paddle.layer.img_pool(input=layer, pool_size=7, stride=7, pool_type=paddle.pooling.Avg())
    return layer

def network():
    # -1 ,2048 
    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2048*5))

    layer = paddle.layer.fc(input=x, size=28*28*3, act=paddle.activation.Linear())
    #layer = resnet_cifar10(layer)   #1*1*64
    layer = simple_cnn(layer)

    sliced_feature = paddle.layer.block_expand(
            input=layer,
            num_channels=64,
            stride_x=1,
            stride_y=1,
            block_x=1,
            block_y=1)

    gru_forward = paddle.networks.simple_gru(input=sliced_feature, size=128, act=paddle.activation.Relu())
    gru_backward = paddle.networks.simple_gru(input=sliced_feature, size=128, act=paddle.activation.Relu(), reverse=True)

    output = paddle.layer.fc(input=[gru_forward,gru_backward], size=class_dim, act=paddle.activation.Softmax())

    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(3))

    cost = paddle.layer.classification_cost(input=output, label=y)
    parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(
        learning_rate=2e-3,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))
    return cost, parameters, adam_optimizer, output

 
def reader_get_image_and_label(isTrain=True):
    def reader():
        if TEST:
            training_data, validation_data, testing_data = load_data_test()
        else:
            training_data, validation_data, testing_data = load_data() 

        if isTrain:
            datalist = training_data
        else:
            datalist = validation_data
        
        nozero_list = []
        
        for data in datalist:
            # data = random.choice(training_data)
            batch_data = np.zeros((2048,5))    
            if TEST:
                v_data = np.random.random(data["shape"])
            else:  
                if isTrain:     
                    v_data = np.load(os.path.join(data_path,"training", "%s.pkl"%data["id"]))
                else:
                    v_data = np.load(os.path.join(data_path,"validation", "%s.pkl"%data["id"]))
            w = v_data.shape[0]
            label = np.zeros([w], dtype=np.int)

            # 先填充2帧
            for i in range(2):
                _data = np.reshape(v_data[i], (2048,1))
                np.append(batch_data[:, 1:], _data, axis=1)

            for annotations in data["data"]:
                segment = annotations['segment']
                for i in range(int(segment[0]),int(segment[1]+1)):
                    label[i] += 1

            for i in range(len(label)):
                if i+2 >= w:
                    _data = np.zeros((2048,1))
                else:
                    _data = np.reshape(v_data[i+2], (2048,1))
                np.append(batch_data[:, 1:], _data, axis=1)

                _data = np.ravel(batch_data)
                if label[i]>0:
                    nozero_list.append([_data, label[i]])

                yield _data, label[i]

                if len(nozero_list)>0:
                    v = random.choice(nozero_list)
                    yield v[0], v[1]
            print("has trained: %s, shape: %s"%(data["id"], v_data.shape))
    return reader

def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id>0 and event.batch_id % 10 == 0:
            print("\nPass %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics) )
            if not TEST:
                with open(param_file, 'wb') as f:
                    print("saveing parameters ...")
                    parameters.to_tar(f)
                    shutil.copy(param_file, param_file_bak)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
#    if isinstance(event, paddle.event.EndPass):
#         result = trainer.test(
#             reader=paddle.batch(
#                 paddle.dataset.cifar.test10(), batch_size=128),
#             feeding=feeding)
#         print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
        
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=1) 
paddle.init(use_gpu=False)  
print("get network ...")
cost, parameters, adam_optimizer,  output= network()
print('set reader ...')
train_reader = paddle.batch(reader_get_image_and_label(), batch_size=256)
feeding={'x': 0, 'y': 1}

if os.path.exists(param_file):
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(param_file)
    print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
    print("loading parameters ...")
    parameters = paddle.parameters.Parameters.from_tar(open(param_file,"rb"))
    
trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=adam_optimizer)
    
print("start train ...")
trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=1)

def getTestData(testFileid):
    if TEST:
        v_data = np.random.random((random.randint(20,30), 2048))
    else:
        v_data = np.load(os.path.join(data_path,"testing", "%s.pkl"%testFileid))

    data = []
    batch_data = np.zeros((2048,5), dtype=np.float)  
    for i in range(2):
        _data = np.reshape(v_data[i], (2048,1))
        np.append(batch_data[:, 1:], _data, axis=1)
    w = v_data.shape[0]
    label = np.zeros([w], dtype=np.int)
    for i in range(w):
        if i+2 >= w:
            _data = np.zeros((2048,1))
        else:
            _data = np.reshape(v_data[i+2], (2048,1))
        np.append(batch_data[:, 1:], _data, axis=1)
        _data1 = np.ravel(batch_data)
        data.append((_data1,))
    return data

# 转换分类到段
# 0000111111111111100000
# 0000000000001111111110
def conv_to_segment(probs):
    sort_probs = np.argsort(-probs)
    value_probs = [v[0] for v in sort_probs]
    items = []
    score = 0
    start = None
    end = None
    #先找重合的块：
    for i, v in enumerate(value_probs):  
        # 如果 v == 2 或者往后5秒内还有>0的，都算
        _continue = False
        if start != None and w > i:
            for j in range(min(w-i,5)):
                if value_probs[i+j+1]>0:
                    _continue = True
                    break
        if v==2 or _continue :
            if start==None:
                start = i
            else:
                end = i
            if v > 0:
                score += probs[i][v]
                value_probs[i] = value_probs[i]-1
            else:
                score += probs[i][1]
        else:
            if start!=None:
                seg_value ={}
                seg_value["score"]=score/(end-start)
                seg_value["segment"]=[start, end]
                start = None
                end = None
                items.append(seg_value)
    if (start != None) and (end !=None):
        seg_value ={}
        seg_value["score"]=score/(end-start)
        seg_value["segment"]=[start, end]
        start = None
        end = None
        items.append(seg_value)

    # 再来找正常的块
    score = 0
    start = None
    end = None
    #先找重合的块：
    for i, v in enumerate(value_probs):  
        # 如果 v == 1 或者往后5秒内还有>0的，都算
        _continue = False
        if start != None and w > i:
            for j in range(min(w-i,5)):
                if value_probs[i+j+1]>0:
                    _continue = True
                    break
        if v==1 or _continue :
            if start==None:
                start = i
            else:
                end = i
            if v > 0:
                score += probs[i][v]
            else:
                score += probs[i][1]
        else:
            if start!=None:
                seg_value ={}
                seg_value["score"]=score/(end-start)
                seg_value["segment"]=[start, end]
                start = None
                end = None
                items.append(seg_value)
    if (start != None) and (end !=None):
        seg_value ={}
        seg_value["score"]=score/(end-start)
        seg_value["segment"]=[start, end]
        start = None
        end = None
        items.append(seg_value)    
    return items

def test(output, parameters):
    if TEST:
        training_data, validation_data, testing_data = load_data_test()
    else:
        training_data, validation_data, testing_data = load_data() 

    result={}
    result["version"]="VERSION 1.0"
    result["results"]={}

    for data in testing_data:
        data_id = data["id"]
        data = getTestData(data_id)  
        w = len(data)
        probs = paddle.infer(output_layer=output, parameters=parameters, input=data)
        items = conv_to_segment(probs)
        result["results"][data_id] = items
        print(json.dumps(result))
        break

    return result

result = test(output, parameters)

print(result)