# coding=utf-8
#!/usr/bin/python

#准备数据

import numpy as np
import os
import json
import time
import random

home = os.path.dirname(__file__)
data_path = os.path.join(home,"data")
model_path = os.path.join(home,"model")
pre_data_path_0 = os.path.join(home,"data_0")
pre_data_path_1 = os.path.join(home,"data_1")

training_path = os.path.join(data_path,"training","image_resnet50_feature")
validation_path = os.path.join(data_path,"validation","image_resnet50_feature")
testing_path = os.path.join(data_path,"testing","image_resnet50_feature")

block_size = 16

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

def pre_data():
    if not os.path.exists(pre_data_path_0): os.mkdir(pre_data_path_0)
    if not os.path.exists(pre_data_path_1): os.mkdir(pre_data_path_1)
    print("clear files ...")
    for _file in os.listdir(pre_data_path_0):
        os.remove(os.path.join(pre_data_path_0,_file))
    for _file in os.listdir(pre_data_path_1):
        os.remove(os.path.join(pre_data_path_1,_file))
    print("cleared")
    size = len(training_data)
    for i, data in enumerate(training_data):
        print("reading %s/%s %s.pkl"%(i,size,data["id"]))
        v_data = np.load(os.path.join(training_path, "%s.pkl"%data["id"]))  

        #生成精彩和非精彩分类
        w = v_data.shape[0]
        label = [0 for _ in range(w)]
        for annotations in data["data"]:
            segment = annotations['segment']
            for i in range(int(segment[0]),min(w,int(segment[1]+1))):
                label[i] = 1
        
        #上一步的值
        prev_status=-1
        prev_value =1.
        for i in range(block_size, w):
            label_sum = sum(label[i-block_size:i])
            if label_sum == block_size:
                if prev_status != 1:
                    prev_value = 0
                    prev_status = 1
                else:
                    prev_value += 1
                # if i % int(round(prev_value)) != 0: continue
                if prev_value >= 8 : continue
                _file = os.path.join(pre_data_path_1,"%s_%d.pkl"%(data["id"],i)) 
                if os.path.exists(_file): 
                    continue
            elif label_sum == 0:
                if prev_status != 0:
                    prev_value = 0
                    prev_status = 0
                else:
                    prev_value += 1
                # if i % int(round(prev_value)) != 0: continue
                if prev_value >= 8 : continue
                _file = os.path.join(pre_data_path_0,"%s_%d.pkl"%(data["id"],i)) 
                if os.path.exists(_file): 
                    continue
            else: continue
            _data = np.stack([v_data[j] for j in range(i-block_size,i)])
            np.save(open(_file,"wb"), _data)

        prev_status=-1
        prev_value =1.
        for i in range(w, block_size, -1):
            label_sum = sum(label[i-block_size:i])
            if label_sum == block_size:
                if prev_status != 1:
                    prev_value = 0.5
                    prev_status = 1
                else:
                    prev_value += 0.2
                if i % int(round(prev_value)) != 0: continue
                _file = os.path.join(pre_data_path_1,"%s_%d.pkl"%(data["id"],i)) 
                if os.path.exists(_file): 
                    continue
            elif label_sum == 0:
                if prev_status != 0:
                    prev_value = 0.5
                    prev_status = 0
                else:
                    prev_value += 0.2
                if i % int(round(prev_value)) != 0: continue
                _file = os.path.join(pre_data_path_0,"%s_%d.pkl"%(data["id"],i)) 
                if os.path.exists(_file): 
                    continue
            else: continue
            _data = np.stack([v_data[j] for j in range(i-block_size,i)])
            np.save(open(_file,"wb"), _data)

if __name__ == '__main__':
    status={}
    status["starttime"] = time.time()
    pre_data()
    print time.time() - status["starttime"]