# n-gram 找出重要词

import os
import jieba
import re
import math

def generate_ngrams(s, n=2):
    # 清理句子
    s = s.lower()
    # 利用结巴分词进行分词
    words = jieba.lcut(s)
    if ' ' in words:
        words.remove(' ')
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def calc(sentence, ngrams, ngrams_length):
    p = 0
    ngrams_list = generate_ngrams(sentence)
    for ngram in ngrams_list:
        if ngram in ngrams:
            p += math.log(ngrams[ngram]/ngrams_length)
        else:
            p += math.log(1e-9/ngrams_length)
    return p

def main():
    curr_path = os.path.dirname(os.path.realpath(__file__))
    txt_file = os.path.join(curr_path,'../12-应用.md')

    ngrams = {}
    sentences = open(txt_file).readlines()
    for s in sentences:
        s = s.strip()
        if s.startswith('$'): continue 
        if len(s)<10: continue 
        for ngram in generate_ngrams(s):
            if ngram in ngrams:
                ngrams[ngram]+=1
            else:
                ngrams[ngram]=1

    ngrams_length = 0
    for ngram in ngrams:
        ngrams_length += ngrams[ngram] 

    for w in sorted(ngrams, key=ngrams.get, reverse=True)[:100]:
        print(w, ngrams[w])

    strs=["神经语言模型网络n-gram", "语言模型n-gram神经网络", "神经n-gram网络语言模型"]
    # 概率取对数
    for _str in strs:
        print(_str,calc(_str,ngrams,ngrams_length))

if __name__ == "__main__":
    main()
