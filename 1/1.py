# coding=utf-8
'''
一个最简单的神经网络，分类问题
'''

import random
import matplotlib.pyplot as plt

# 数据，输入 x1,x2 ，如果 x2 大于 200 数据归为 1，否则为 -1
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
        train_x.append([x1, x2])
        train_y.append(y)
    return train_x, train_y

# 权重,为了避免计算为0，初始化一个随机值
W = [0, 0]
for i in range(len(W)):
    W[i] = random.uniform(-1, 1)
# 偏置量
b = 0

# 分类模型，根据这个模型来动态调整权重W的值
def reduce(x):
    sum = 0.0
    for i in range(len(x)):
        sum += x[i] * W[i]
    sum += b
    if sum > 0:
        return 1
    else:
        return -1

# 学习算法
def train_step(x, y, rate=0.01):
    global W, b
    guess = reduce(x)
    error = y - guess
    for i in range(len(x)):
        W[i] += rate * x[i] * error
    b += rate * error

# 学习一批长度
batch_size = 50000

def train():
    for step in range(100):
        train_x, train_y = batch(batch_size)
        for i in range(batch_size):
            train_step(train_x[i], train_y[i])
        acc, test_x, test_r = test(100)
        print(step, acc, W, b)
        if step == 0:
            plotbatch(train_x, train_y, 100)
        if (step + 1) % 20 == 0:
            plottest(test_x, test_r)

# 测试准确率
def test(batch_size):
    test_x, test_y = batch(batch_size)
    test_r = []
    acc = 0.0
    for i in range(batch_size):
        if reduce(test_x[i]) == test_y[i]:
            acc += 1
            test_r.append(1)
        else:
            test_r.append(0)
    return acc / batch_size, test_x, test_r

# 绘制数据
def plotbatch(train_x, train_y, batch_size=100):
    x1_data = []
    y1_data = []
    x2_data = []
    y2_data = []
    for i in range(batch_size):
        if train_y[i] == 1:
            x1_data.append(train_x[i][0])
            y1_data.append(train_x[i][1])
        else:
            x2_data.append(train_x[i][0])
            y2_data.append(train_x[i][1])
    plt.plot(x1_data, y1_data, 'ro')
    plt.plot(x2_data, y2_data, 'bo')
    plt.axis([0, 500, 0, 500])
    plt.show()

# 绘制测试结果
def plottest(test_x, test_r):
    x1_data = []
    y1_data = []
    x2_data = []
    y2_data = []
    for i in range(len(test_r)):
        if test_r[i]:
            x1_data.append(test_x[i][0])
            y1_data.append(test_x[i][1])
        else:
            x2_data.append(test_x[i][0])
            y2_data.append(test_x[i][1])
    plt.plot(x1_data, y1_data, 'g+')
    plt.plot(x2_data, y2_data, 'rx')
    plt.axis([0, 500, 0, 500])

    x1 = 0
    y1 = - (W[0] * x1 + b) / W[1]
    x2 = 500
    y2 = - (W[0] * x1 + b) / W[1]
    plt.plot([x1, x2], [y1, y2])

    plt.show()

if __name__ == '__main__':
    train()
