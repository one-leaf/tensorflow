#!/usr/bin/python3

import jieba
import os
import numpy as np

# declare_list 海关分类
# hsmodel_list 用户数据

def clearData(declare_list, hsmodel_list):
    # 如果分类的最后一项是''，则删除该项,重复2次
    if declare_list[-1].strip()=='':
        declare_list=declare_list[:-1]
    if declare_list[-1].strip()=='':
        declare_list=declare_list[:-1]

    # 去除 品牌类型，出口享惠情况 的干扰，这两个值都应该是 [0,1,2,3,4]
    # 品牌类型，出口享惠情况 ，直接忽略
    if len(declare_list)>=2:
        if "1品牌类型" in declare_list:
            declare_list.remove("1品牌类型")
        if "2出口享惠情况" in declare_list:
            declare_list.remove("2出口享惠情况")
    # 如果企业数据，前2项的内容不在 [0,1,2,3,4] 范围内，则忽略规范申报的2项 
    if len(hsmodel_list)>=3 and hsmodel_list[0] in ['0','1','2','3','4'] and (hsmodel_list[1] in ['0','1','2','3','4','']):
        hsmodel_list = hsmodel_list[2:] 

    # 清理栏位,去除空格
    for i, hsmodel in enumerate(hsmodel_list):
        hsmodel_list[i] = hsmodel.strip()

    for i, declare in enumerate(declare_list):
        clear = declare.strip()
        # 清除前面的数字
        k_clear = ""
        check_stop = False
        for c in clear:
            # 去除规范申报名称前面的序号
            if not check_stop and c in ['0','1','2','3','4','5','6','7','8','9',':']: continue
            k_clear = k_clear + c
            check_stop = True
        declare_list[i] = k_clear

    if "GTIN" in declare_list and "无GTIN" not in hsmodel_list : declare_list.remove("GTIN")
    if "CAS" in declare_list and "无CAS" not in hsmodel_list: declare_list.remove("CAS")

    return declare_list, hsmodel_list

# 分词 sentence 句子
def splitSentence(sentence):
    data=list(jieba.cut(sentence, cut_all=False))
    output=[]
    for i, s in enumerate(data):
        v = s.strip()
        if v == "" : continue

        # 如果字段中包含数字，直接拆开为单个字母，因为合起来的数字对样本无意义
        is_include_number = False
        for c in v:
            if c in ['0','1','2','3','4','5','6','7','8','9']:
                is_include_number = True
                break
        
        if is_include_number:
            for c in v:
                output.append(c)
        else:
            output.append(v)
    return output

# 读取words
def load_words():
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")
    words_filename = os.path.join(data_dir,"words.txt")   
    word_list = []
    with open(words_filename,"r", encoding='UTF-8') as f:
        for line in f:
            if line!="":
                word_list.append(line.strip())
    return word_list

dictionary=None

# 根据wold建立索引, 0为UNK 
def get_word_dict():
    global dictionary
    if dictionary!=None:
        return dictionary

    print("Loading words.txt ...")
    words = load_words()
    dictionary = dict()
    for i, word in enumerate(words):
        dictionary[word]=i
    del words
    return dictionary

# 加载word向量
embed=[]
def get_word_embed():
    global embed
    if len(embed)>0: 
        return embed
    print("Loading embedding.npy ...")
    curr_dir = os.path.dirname(__file__)
    data_dir = os.path.join(curr_dir, "data")
    embed_filename = os.path.join(data_dir,"embedding.npy")   
    embed = np.load(embed_filename)
    return embed

# 由word转word向量
def getWord2Vec(word):
    word_dict = get_word_dict()
    word_embed = get_word_embed()
    if word in word_dict:
        return word_embed[word_dict[word]]
    else:
        return word_embed[0]

