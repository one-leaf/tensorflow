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

model = __import__('7')
home = os.path.dirname(__file__)
data_path = model.data_path
model_path = model.model_path
cls_param_file = model.cls_param_file
result_json_file = os.path.join(model_path,"ai.json.zip")
out_dir = os.path.join(model_path, "out")

# home = "/home/kesci/work/"
# data_path = "/mnt/BROAD-datasets/video/"
# param_file = "/home/kesci/work/param2.data"
# param_file_bak = "/home/kesci/work/param2.data.bak"
# result_json_file = "/home/kesci/work/ai2.json"

if not os.path.exists(model_path): os.mkdir(model_path)
if not os.path.exists(out_dir): os.mkdir(out_dir)

      
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=2) 
paddle.init(use_gpu=True, trainer_count=1)
print("get network ...")
cost, adam_optimizer, net_class_fc  = model.network(drop=False)

# 预测时需要读取模型
(mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(cls_param_file)
print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
print("loading parameters ...", cls_param_file)
paddle_parameters = paddle.parameters.Parameters.from_tar(open(cls_param_file,"rb"))
    
def test():
    items = []
    inferer = paddle.inference.Inference(output_layer=net_class_fc, parameters=paddle_parameters)

    for i, data_info in enumerate(model.training_data):    
    # for i, data_info in enumerate(model.validation_data):    
        data_id = data_info["id"]
        v_data = np.load(os.path.join(model.training_path, "%s.pkl"%data_id))
        # v_data = np.load(os.path.join(data_path, "validation", "%s.pkl"%data_id))

        # 得到直观分布图
        w = v_data.shape[0]
        print "读取数据:", data_id, v_data.shape
        label = np.zeros([w], dtype=np.int)

        for annotations in data_info["data"]:
            segment = annotations['segment']
            start = int(round(segment[0]))
            end = int(round(segment[1]))
            print(start, end)
            for i in range(start-model.block_size, end+1):
                if i<0 or i>=w: continue
                if i+model.block_size>start and i<=start: 
                    label[i] = 1
                elif i+model.block_size>end and i<=end:
                    label[i] = 3
                elif label[i] == 0:
                    label[i] = 2

        # print label
        save_file = os.path.join(out_dir,data_id)
        if not os.path.exists(save_file):
            _data=[]
            for i in range(w-model.block_size):
                _data.append(v_data[i:i+model.block_size])
                if len(_data) == model.train_size:
                    print "正确目标：",i-model.train_size,"-",i+1
                    print label[i-model.train_size+1:i+1]                    
                    probs = inferer.infer(input=[(_data,)])
                    print probs[:,1]

                    # 预测当前方块是否是精华或非精华                    
                    sort = np.argsort(-probs)
                    value_probs = sort[:,0]
                    print  "判断分类",value_probs
                    _data=[]
                    if raw_input("==========================================================================="): pass
            #     all_values.append(probs)
            #     sys.stdout.write(".")
            #     sys.stdout.flush()           
                
            # if w%batch_size != 0:
            #     _data = data[count*batch_size:]
            #     probs = inferer.infer(input=_data)
            #     all_values.append(probs)
            #     sys.stdout.write('.')
            #     sys.stdout.flush() 
        
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
