#!/usr/bin/python3
# 5 建立分类训练数据集

import os
import json
from utils import splitSentence
import hashlib

def main():
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")
    # 定义句子和分词
    dataset_filename = os.path.join(data_dir,"dataset.txt")
    dataset_file = open(dataset_filename, "w", encoding="UTF-8") 
    check_same = set()
    fname = 'decitem.txt'
    with open(os.path.join(data_dir,fname),encoding="UTF-8") as f:
        for count, line in enumerate(f): 
            if count%10000==0: print(count,len(check_same)) 

            # 去重复
            hl = hashlib.md5()
            hl.update(line[:(line.index("\"id\""))].encode(encoding='utf-8'))
            hl = hl.hexdigest()
            if hl in check_same:
                continue
            else:
                check_same.add(hl)

            row = json.loads(line)
            dataset=[[],[]]
            for k in row:
                if k=='id': continue

                # # 忽略掉值长度为1的，太少了不足以判断，只会干扰学习
                # v = row[k]
                # if len(v)<=1: continue

                dataset[0].append(splitSentence(k))
                dataset[1].append(splitSentence(row[k]))

            json_out = json.dumps(dataset, ensure_ascii=False)
            dataset_file.write(json_out+'\n')
    dataset_file.close()

if __name__ == '__main__':
    main()