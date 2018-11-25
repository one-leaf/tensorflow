# 绘制似然函数
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math

def draw(n):
    # 投掷为正面的概率
    p_list = [0.1,0.3,0.5,2/3,0.8]
    x_data = np.linspace(0,1,n)
    y_nk = [math.factorial(n)/(math.factorial(k) * math.factorial(n-k)) for k in range(n)]

    c = iter(cm.rainbow(np.linspace(0, 0.8, len(p_list))))
    for p in p_list:
        y_value = [p**k*(1-p)**(n-k) for k in range(n)]
        y_value = [y_nk[k]*y_value[k] for k in range(n)]
        plt.plot(x_data, y_value, '-', color=next(c), label='p=%s'%p)

    # 绘制似然函数曲线
    x = np.linspace(0,1,n)
    y = x*x*(1-x)
    plt.plot(x,y,'r-', label='L')
    plt.grid(True)
    plt.legend()

draw(100)
plt.show()
