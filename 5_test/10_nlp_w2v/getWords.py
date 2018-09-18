#!/usr/bin/python3
'''
4. 提取单词 小于100次出现的词汇忽略
'''
import os
import json


def main():
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")
    sentence_filename = os.path.join(data_dir,"sentence.txt")   
    words_filename = os.path.join(data_dir,"words.txt")   
    word_lists={}
    with open(os.path.join(data_dir,sentence_filename),encoding="UTF-8") as f:
        for count, line in enumerate(f): 
            if count%10000==0: print(count)  
            words = line.split(" ")
            for word in words:
                word=word.strip()
                if word=="": continue
                if word in word_lists:
                    word_lists[word] += 1
                else:
                    word_lists[word] = 1
    count = len(word_lists)
    
    i=0
    words=[]
    for key, value in word_lists.items():
        if value<50: continue
        i+=1
        print ("%s %s: %s" % (i, key, value))
        words.append(key)
        
    words_file = open(words_filename, "w", encoding="UTF-8")
    # 第一个是 UNKNOWN 未知
    words_file.write("UNK\n")
    words.sort()
    for word in words:    
        words_file.write(word+"\n")
    words_file.close()


if __name__ == '__main__':
    main()