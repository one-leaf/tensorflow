#!/usr/bin/python3
'''
3. 提取文本，并分词
'''

import os
import json
from utils import splitSentence

def main():
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")
    # 定义句子和分词
    words_filename = os.path.join(data_dir,"sentence.txt")
    words_file = open(words_filename, "w", encoding="UTF-8") 
    fname = 'decitem.txt'

    with open(os.path.join(data_dir,fname),encoding="UTF-8") as f:
        for count, line in enumerate(f): 
            if count%10000==0: print(count)  
            row = json.loads(line)
            for k in row:
                if k=='id': continue
                v = row[k]
                mstr = k + " " + v + " " + k
                output = splitSentence(mstr)
                readline = ' '.join(output)+'\n'
                words_file.write(readline)
    words_file.close()

if __name__ == '__main__':
    main()