# sigmoid 和 tanh 对比
import math
import matplotlib.pyplot as plt
import numpy as np
 
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
  
x = np.linspace(-6, 6, 100)
y = sigmoid(x)
tanh = 2*sigmoid(2*x) - 1

# 设置坐标轴
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

plt.grid(True)
plt.plot(x,y,label="Sigmoid",color = "b")
plt.plot(x,y*(1-y),label="Sigmoid Derivative",color = "g")


plt.plot(x,tanh,label="Tanh", color = "r")
plt.plot(x,1-np.power(tanh,2),label="Tanh Derivative", color = "y")

plt.legend()
plt.show()
