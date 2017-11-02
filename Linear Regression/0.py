# 线性回归

import numpy as np
import random
import matplotlib.pyplot as plt

# 获得数据 / 10 只是让数据好看一点
def getBatch(batchSize,start=None):
    if start==None:    
        start = random.randint(1, 10000)
    n = np.linspace(start, start+batchSize, batchSize, endpoint=True).reshape((batchSize,1)) / 10
    x = np.sin(n)
    y = np.cos(n)
    return x,y,n

if __name__ == '__main__':
    batch_x, batch_y, batch_n= getBatch(200, 0)
    plt.plot(batch_n, batch_x, 'r', label="x")
    plt.plot(batch_n, batch_y, 'b', label="y")
    plt.legend(loc='lower right')
    plt.show()