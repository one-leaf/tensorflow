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

class_dim = 2 # 分类 0，背景， 1，精彩
box_dim = 2 # 偏移，左，右
train_size = 256 # 学习的关键帧长度
buf_size = 8192
batch_size = 4
block_size = 64


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
    x = paddle.layer.data(name='x', height=train_size, width=2048, type=paddle.data_type.dense_vector(train_size*2048))  
    # x_emb = paddle.layer.embedding(input=x, size=train_size*2048)  # emb一定要interger数据？

    c = paddle.layer.data(name='c', type=paddle.data_type.integer_value_sequence(class_dim))
    # c_emb = paddle.layer.embedding(input=c, size=train_size)

    b = paddle.layer.data(name='b', type=paddle.data_type.dense_vector_sequence(box_dim))
    # b_emb = paddle.layer.embedding(input=b, size=train_size)

    main_nets = []
    net = cnn2(x,  3,  1, 64, 1, 1)
    net = cnn2(net, 3, 64, 64, 1, 1)
    main_nets.append(net)
    net = cnn2(net, 3, 64, 64, 1, 1)
    net = cnn2(net, 3, 64, 64, 1, 1)
    main_nets.append(net)
    net = cnn2(net, 3, 64, 64, 1, 1)
    net = cnn2(net, 3, 64, 64, 1, 1)
    main_nets.append(net)
    net = cnn2(net, 3, 64, 64, 1, 1)
    net = cnn2(net, 3, 64, 64, 1, 1)
    main_nets.append(net)  
    net = cnn2(net, 3, 64, 64, 1, 1)
    net = cnn2(net, 3, 64, 64, 1, 1)
    main_nets.append(net)  
 
    blocks = []
    for i  in range(len(main_nets)):
        main_net = main_nets[i]
        block_expand = paddle.layer.block_expand(input= main_net, num_channels=64, stride_x=1, stride_y=1, block_x=main_net.width, block_y=1)
        blocks.append(block_expand)

    costs=[]
    net_class_fc = paddle.layer.fc(input=blocks, size=class_dim, act=paddle.activation.Softmax())
    net_box_fc = paddle.layer.fc(input=blocks, size=class_dim, act=paddle.activation.Tanh())
    cost_class = paddle.layer.classification_cost(input=net_class_fc, label=c)
    cost_box = paddle.layer.square_error_cost(input=net_box_fc, label=b)
    costs.append(cost_class)
    costs.append(cost_box)
    
    parameters = paddle.parameters.create(costs)
    adam_optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    return costs, parameters, adam_optimizer, net_class_fc, net_box_fc 
      
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=2) 
paddle.init(use_gpu=True, trainer_count=1)
print("get network ...")
cost, paddle_parameters, adam_optimizer, net_class_fc, net_box_fc = network()

# 预测时需要读取模型
(mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(param_file)
print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
print("loading parameters ...")
paddle_parameters = paddle.parameters.Parameters.from_tar(open(param_file,"rb"))
    

def getTestData(testFileid):
    v_data = np.load(os.path.join(data_path,"validation", "%s.pkl"%testFileid))
    data = []
    batch_data = [np.zeros(2048) for _ in range(train_size)]   
    w = v_data.shape[0]
    label = np.zeros([w], dtype=np.int)
    for i in range(w):
        batch_data.append(v_data[i])
        batch_data.pop(0)
        if i>0 and i%train_size==0:
            data.append((batch_data,))
    return data, len(w)

def test():
    items = []
    _, validation_data, _ = load_data("validation") 
    size = len(validation_data)
    inferer = paddle.inference.Inference(output_layer=[net_class_fc,net_box_fc], parameters=paddle_parameters)

    for i, data_info in enumerate(validation_data):       
        data_id = data_info["id"]

        data, label_size = getTestData(data_id)  
        
        w = len(data)
        print("\nstart infer: %s / %s  %s size %s"%(i, size, data_id, w))
        
        all_values=[]
        batch_size = 1
        count = w // batch_size
        print("need infer count:", count)

        label = np.zeros([label_size], dtype=np.int)

        for annotations in data_info["data"]:
            segment = annotations['segment']
            for i in range(int(segment[0]),int(segment[1]+1)):
                label[i] += 1

        save_file = os.path.join(out_dir,data_id)
        if not os.path.exists(save_file):

            for i in range(count):
                _data = data[i*batch_size:(i+1)*batch_size]
                probs = inferer.infer(input=_data,field=["value","value"])

                probs_class = probs[0]
                probs_box = probs[1]

                sort_probs = np.argsort(-probs_class)
                value_probs = sort_probs[:,0]
                # print(probs_class)
                # print(sort_probs)
                print(value_probs)
                print(label[0:train_size])
                return
                all_values.append(probs)
                sys.stdout.write(".")
                sys.stdout.flush()           
                
            if w%batch_size != 0:
                _data = data[count*batch_size:]
                probs = inferer.infer(input=_data)
                all_values.append(probs)
                sys.stdout.write('.')
                sys.stdout.flush() 
        
            _all_values = np.row_stack(all_values)
            np.save(open(save_file,"wb"), _all_values)
        else:
            _all_values = np.load(open(save_file,"rb"))



        # print(label[0:999])

        # value_probs = np.argsort(-_all_values)[:,0]
        # for i,v in enumerate(value_probs):
        #     if _all_values[i][v]>0.99 and v==1:
        #         value_probs[i-train_size+1:i+1] = v
        # print(value_probs[0:999])

        # print(np.argsort(-_all_values)[:,0][0:999])
        # print(np.max(_all_values,axis=1)[0:999])

        # item = conv_to_segment(_all_values)
        item =[]
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
