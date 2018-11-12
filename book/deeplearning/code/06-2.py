import math
import matplotlib.pyplot as plt
import numpy as np
 
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
 
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)
 
x = np.linspace(-10, 10)
y = sigmoid(x)
tanh = 2*sigmoid(2*x) - 1
 
plt.xlim(-11,11)
plt.ylim(-1.1,1.1)
 
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
 
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data',0))
# ax.set_xticks([-10,-5,0,5,10])
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',0))
# ax.set_yticks()
 

plt.xticks([-10,-5,0,5,10])
plt.yticks([-1,-0.5,0.5,1])

plt.plot(x,y,label="Sigmoid",color = "blue")
plt.plot(2*x,tanh,label="Tanh", color = "red")
plt.legend()
plt.show()
