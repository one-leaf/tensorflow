# relu 各函数对比
import math
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(x, a):
    return np.maximum(0,x)+a*np.minimum(0,x)

def softplus(x):
    return np.log(1+np.exp(x))

x = np.linspace(-10, 10, 100)

# 设置坐标轴
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

plt.grid(True)
plt.plot(x, relu(x, 0.1), "r", label="Leaky ReLU")
plt.plot(x, x*sigmoid(1*x), "g", label="Swish")
plt.plot(x, softplus(x), "y", label="softplus")
plt.plot(x, relu(x, 0), "b", label="ReLU")
plt.legend()
plt.show()
