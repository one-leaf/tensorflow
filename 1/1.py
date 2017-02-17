# coding=utf-8
'''
一个最简单的神经网络，分类问题
'''

import random

#数据，输入 x1,x2 ，如果 x2 大于 200 数据归为 1，否则为 -1
def batch(batch_size):
    train_x = []
    train_y = []
    for i in range(batch_size):
        x1 = random.randint(0, 500)
        x2 = random.randint(0, 500)
        if x2 > 200: 
            y = 1
        else:
            y = -1
        train_x.append([x1,x2])
        train_y.append(y)
    return train_x,train_y


#权重
W = []
#由于有输入有两项，初始化一个-1到1之间的随机浮点数给权重
for _ in range(3):
    W.append(random.uniform(-1, 1))

#偏置量
b = 1

#分类模型，根据这个模型来动态调整权重W的值
def reduce(x):
    sum = 0.0
    x1=x+[b]
    for i in range(len(W)):
        sum += x1[i] * W[i] 
    if sum>0:
        return 1
    else:
        return -1

#学习算法
def train_step(x,y,rate=0.01):
    guess = reduce(x)
    error = y - guess
    x1=x+[b]
    for i in range(len(W)):
        W[i] += rate*error*x1[i]

#学习一批长度
batch_size=100000
def train():   
    for step in range(100):
        train_x,train_y=batch(batch_size)
        for i in range(batch_size):
            train_step(train_x[i],train_y[i])
        acc=test(100)    
        print(step,acc,W)    

#测试准确率        
def test(batch_size):
    train_x,train_y=batch(batch_size)
    acc=0.0
    for i in range(batch_size):
        if reduce(train_x[i])==train_y[i]:
            acc += 1
    return acc/batch_size        

if __name__ == '__main__':
    train()
