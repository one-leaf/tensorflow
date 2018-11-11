import matplotlib.pyplot as plt
import numpy as np
import math

sampleNo = 1000

def draw(n):
    # 投掷为正面的概率
    p = 0.3
    s=np.random.binomial(n, p, size=sampleNo)
    plt.hist(s, 50, label='n=%s, p=%s'%(n,p))
    plt.xlim(0, n)
    plt.grid(True)
    plt.legend()
    plt.show()

    p_list = [0.1, 0.3, 0.5, 0.8] 

    # 投掷为正面的次数
    x_data = [k for k in range(n)]

    y_nk = [math.factorial(n)/(math.factorial(k) * math.factorial(n-k)) for k in range(n)]

    y_data=[]
    for p in p_list:
        y_value = [p**k*(1-p)**(n-k) for k in range(n)]
        y_value = [y_nk[k]*y_value[k] for k in range(n)]
        y_data.append(y_value)

    plt.plot(x_data, y_data[0], 'b-', label='n=%s, p=%s'%(n, p_list[0]))
    plt.plot(x_data, y_data[1], 'r-', label='n=%s, p=%s'%(n, p_list[1]))
    plt.plot(x_data, y_data[2], 'g-', label='n=%s, p=%s'%(n, p_list[2]))
    plt.plot(x_data, y_data[3], 'y-', label='n=%s, p=%s'%(n, p_list[3]))

    plt.grid(True)
    plt.legend()
    plt.show()

draw(100)
draw(1000)