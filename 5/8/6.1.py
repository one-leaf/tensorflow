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

model = __import__('6')
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

      
print("paddle init ...")
# paddle.init(use_gpu=False, trainer_count=2) 
paddle.init(use_gpu=True, trainer_count=1)
print("get network ...")
cost, paddle_parameters, adam_optimizer, net_class_fc, net_class_box_fc, net_box_fc = model.network()

# 预测时需要读取模型
(mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(param_file)
print("find param file, modify time: %s file size: %s" % (time.ctime(mtime), size))
print("loading parameters ...")
paddle_parameters = paddle.parameters.Parameters.from_tar(open(param_file,"rb"))
    
def test():
    items = []
    inferer = paddle.inference.Inference(output_layer=[net_class_fc,net_class_box_fc,net_box_fc], parameters=paddle_parameters)

    for i, data_info in enumerate(model.training_data):       
        data_id = data_info["id"]
        v_data = np.load(os.path.join(data_path, "training", "%s.pkl"%data_id))

        # 得到直观分布图
        w = v_data.shape[0]
        label = np.zeros([w], dtype=np.int)
        for annotations in data_info["data"]:
            segment = annotations['segment']
            for i in range(int(segment[0]),int(segment[1]+1)):
                label[i] += 1

        save_file = os.path.join(out_dir,data_id)
        if not os.path.exists(save_file):
            for i, _data in model.read_data(v_data):
                probs = inferer.infer(input=[(_data,)])
                print "正确目标：",label[i-model.train_size:i]

                # 预测当前方块是否是精华或非精华
                probs_class = probs[0: model.train_size]
                sort = np.argsort(-probs_class)
                value_probs = sort[:,0]
                print  "判断分类",value_probs

                # print(probs_class)
                probs_box_class = probs[model.train_size: model.train_size*2]
                has_class = probs_box_class[:,1]
                sort = np.argsort(-has_class)
                print "前五最高：",sort[0:5]
                print "概率如下：",has_class[sort[0:5]]
                probs_net = probs[model.train_size*2:]
                print i-model.train_size, i
                for s in sort[0:5]:
                    if has_class[s]<0.5: break
                    src = model.get_box_point(s)
                    print s, has_class[s], probs_net[s]
                    print "分类坐标：", src
                    print "偏移量：", probs_net[s]*model.train_size
                    fix_src= [max(src[0]+probs_net[s][0]*model.train_size,0),min(src[1]+probs_net[s][1]*model.train_size,model.train_size)]
                    print "预测坐标：", fix_src
                    label2 = np.zeros([model.train_size], dtype=np.int)        
                    for x in range(int(fix_src[0]),int(fix_src[1]+1)):
                        label2[x] = 1
                    label2[int(s)]=8
                    if src[0]>=0:
                        label2[int(src[0])]=7
                    if src[1]<model.train_size:
                        label2[int(src[1])]=9
                    print "预测目标：", label2[0:model.train_size] 

                if raw_input("press any key to continue:"): pass
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
