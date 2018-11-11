import matplotlib.pyplot as plt
import numpy as np
import math

sampleNo = 10000

# 绘制一个高斯分布的区间
# 偏差，均值
mu = 0
# 标准差
sigma = 0.1
# 产生一个符合高斯的分布 
s = np.random.normal(mu, sigma, sampleNo )
plt.hist(s, 50, label='sigma=0.1, mu=0')
plt.grid(True)
plt.legend()
plt.show()

# 绘制正态分布概率密度函数
sigma=[0.1, 0.2, 1, 6]
sig_list = [math.sqrt(s) for s in sigma]  # 方差

x_data = np.linspace(-10, 10, sampleNo)
y_data = []
for sig in sig_list:
    y_sig = [np.exp(-(x - mu) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig) for x in x_data]
    y_data.append(y_sig)

plt.plot(x_data, y_data[0], "r-", linewidth=1, label="sigma=%s"%sigma[0])
plt.plot(x_data, y_data[1], "g-", linewidth=1, label="sigma=%s"%sigma[1])
plt.plot(x_data, y_data[2], "b-", linewidth=1, label="sigma=%s"%sigma[2])
plt.plot(x_data, y_data[3], "y-", linewidth=1, label="sigma=%s"%sigma[3])

plt.grid(True)
plt.legend()
plt.show()
