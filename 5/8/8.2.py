## DEBUG 用 ##

# coding=utf-8
#!/sbin/python

import numpy as np
import json
import os
import random
import sys
import time
import shutil 
import logging
import gc
import commands, re  
import pickle

# home = "/home/kesci/work/"
# model_path = os.path.join(home,"model")
home = os.path.dirname(__file__)
data_path = os.path.join(home,"data")
result_json_file = os.path.join(model_path,"ai.json")
out_dir = os.path.join(model_path, "out")
if not os.path.exists(model_path): os.mkdir(model_path)
if not os.path.exists(out_dir): os.mkdir(out_dir)
np.set_printoptions(threshold=np.inf)

data_path = "/mnt/BROAD-datasets/video/"
training_path = os.path.join(data_path,"training","image_resnet50_feature")
validation_path = os.path.join(data_path,"validation","image_resnet50_feature")
testing_path = os.path.join(data_path,"testing","image_resnet50_feature")

block_size = 2
mark_length = 6 # 标记

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

training_data, validation_data, testing_data = load_data()

def conv_to_segment(probs, minsec=32, isDebug=False):
    sort_probs = np.argsort(-probs)
#     print(sort_probs)
    v = sort_probs[:,0]
    w=len(v)
    items = []
    if isDebug: print(v)
    start = None
    end = None
    p=0
    while True:
        zerocount=0
        # 如果 剩下的区块小于最小区块，就不再扫描了
        if (w-p)<minsec: break
        for i in range(p, w):
            if v[i]==0:
                zerocount+=1
            
            # 先按2去找start,找到2后，统计后面50个块中是否含有30个有效，如果是，确定开始位置
            if start==None and v[i]==2:
                _onecount=0
                for j in range(i+1,i+50):
                    if j<w and (v[j]==1 or v[j]==2):
                        _onecount+=1
                if _onecount>30:
                    start=i
                else:
                    continue

            # 继续按1去找start, 如果后面20个以内，包括了至少3个2，忽略，以后面的2为准；否则继续找30个1
            if start==None and v[i]==1:
                _twocount=0
                for j in range(i+1,i+20):
                    if j<w and v[j]==2:
                        _twocount+=1
                if _twocount>3:
                    continue
                    
                _onecount=0                
                for j in range(i+1,i+50):
                    if j<w and v[j]==1:
                        _onecount+=1
                if _onecount>30:
                    start=i
                else:
                    continue

            # 如果找到开始位置了，搜索后面的结尾，忽略交叉的，这一个实在太难检测
            
            if start!=None:
                # 先按3搜索，如果搜索不到，再按0
                _zcount=0
                _tcount=0
                for j in range(start+1,w):
                    # 统计连续为0个个数
                    if v[j]==0:
                        _zcount+=1
                    else:
                        _zcount=0
                    
                    if v[j]==3 or v[j]==1:
                        end=j
                        
                    # 如果碰到开始2了，但后面没有3，停止
                    if v[j]==2:
                        _threecount=0
                        for k in range(j+1,j+20):
                            if k<w and v[k]==3:
                                 _threecount+=1
                        if _threecount>0:
                            continue 
                        if end!=None and end-start>minsec:
                            break
                    
                    # 如果后面10位内还包含了3，忽略，以后面的为准
                    _threecount=0
                    for k in range(i+1,i+10):
                        if k<w and v[k]==3:
                             _threecount+=1
                    if _threecount>0:
                        continue 
                    else:
                        # 如果前面10位包括了至少4个3，以这个为end
                        _threecount=0
                        for k in range(j-10,j+1):
                            if k<w and k>0 and v[k]==3:
                                 _threecount+=1
                        if _threecount>4:
                             break
                            
                    # 如果连续0超过20个，且找到了end，且长度超过最低长度，寻找结束
                    if _zcount>20 and end!=None and end-start>minsec:
                        break
                            
                    # 如果连续0超过了30个，不管有没有找到也结束，说明开始选的有错误        
                    if _zcount>30:
                        break

                if start!=None and end!=None and end-start<minsec:
                    end=None
                    
                if start!=None and end!=None:
                    items.append((start,end))
                    
                    #将尾端的结束标记抹去
                    for j in range(start,end+1):
                        if v[j]==3:
                            v[j]=1
                    
                    # 检查后1/2区间是否也包含了连续的2，如果包含3以上的连续2，则下一次定位按2开始
                    _twocount = 0
                    _twopoint = 0
                    for j in range(start+(end-start)//2,end):
                        if v[j]==2:
                            _twocount+=1
                            if _twopoint==0:
                                _twopoint=j
                        else:
                            _twocount=0
                            
                    if _twocount>3:
                        p=_twopoint
                        start=None
                        end=None
                        break    
                    
                if isDebug: print(start,end+1)
                if isDebug: print(v)
                if end!=None:
                    p=end+1
                else:
                    p=i+1
                start=None
                end=None
                break
            else:
                p=i+1
                
    result=[]
    for item in items:
        seg_value ={}
        seg_value["score"]=1
        seg_value["segment"]=item
        result.append(seg_value)          
    return result


def main():
    save_file = os.path.join(out_dir,"validation_4.pkl")
    print("loading...")
    infers = pickle.load(open(save_file,"rb"))
    print("loaded.")
    result={}
    result["version"]="VERSION 1.0"
    result["results"]={}  
    print("start infer...")
    for infer in infers:
        _all_values = infers[infer]
        _all_values = np.stack(_all_values,axis=0)
        
        sort_probs = np.argsort(-_all_values)
        v = sort_probs[:,0]
#         print(v)

        
#         result["results"][infer] = items
#         return
#         print("infered %s count:%s"%(infer,len(items)))
#     json.dump(result, open(result_json_file,"w"))

        for t_data in validation_data:
            if t_data["id"]==infer:                
                v_data = np.load(os.path.join(validation_path, "%s.pkl"%t_data["id"]))  
                w = v.shape[0]
                label = [0 for _ in range(w)]
                for annotations in t_data["data"]:
                    segment = annotations['segment']
                    start = int(round(segment[0]))
                    end = int(round(segment[1]))
                    for i in range(start, end+1):
                        if i<0 or i>=w: continue                  
                        if i>=start and i<=start+mark_length: 
                            label[i] = 2
                        elif i>=end-mark_length and i<=end:
                            label[i] = 3
                        elif label[i] == 0:
                            label[i] = 1                  
                v_val = np.array(label)
                if infer=="310918400":
                    print v_val
                    print v
                    print t_data["data"]
                    print conv_to_segment(_all_values)
                accuracy = (v == v_val).mean()
                print(infer, accuracy)
                break
    print("OK")  
        
if __name__ == '__main__':
    main()