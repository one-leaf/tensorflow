# 绘制高斯分布模型

import matplotlib.pyplot as plt
import numpy as np
import math

sampleNo = 1000

# 绘制一个高斯分布的区间
# 偏差，均值
mu = 0
# 标准差
sigma = 0.1
# 产生一个符合高斯的分布 
s = np.random.normal(mu, sigma, sampleNo )

plt.figure()
plt.hist(s, 50, label='sigma=0.1, mu=0')
plt.grid(True)
plt.legend()

# 绘制正态分布概率分布和密度函数
sigma=[0.1, 0.2, 1, 6]
sig_list = [math.sqrt(s) for s in sigma]  # 方差

x_data = np.linspace(-10, 10, sampleNo)
y_data = []
y_d_data = []
for sig in sig_list:
    # 概率分布
    y_sig = np.exp(-(x_data - mu) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig)
    # 概率密度
    y_sig_d = [math.exp(-(x - mu) ** 2 /(2* sig **2))*(x-mu)/(math.sqrt(2*math.pi)*sig**3) for x in x_data]
    y_data.append(y_sig)
    y_d_data.append(y_sig_d)

plt.figure()
plt.plot(x_data, y_data[0], "r-", linewidth=1, label="sigma=%s"%sigma[0])
plt.plot(x_data, y_data[1], "g-", linewidth=1, label="sigma=%s"%sigma[1])
plt.plot(x_data, y_data[2], "b-", linewidth=1, label="sigma=%s"%sigma[2])
plt.plot(x_data, y_data[3], "y-", linewidth=1, label="sigma=%s"%sigma[3])
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(x_data, y_d_data[0], "r-", linewidth=1, label="d sigma=%s"%sigma[0])
plt.plot(x_data, y_d_data[1], "g-", linewidth=1, label="d sigma=%s"%sigma[1])
plt.plot(x_data, y_d_data[2], "b-", linewidth=1, label="d sigma=%s"%sigma[2])
plt.plot(x_data, y_d_data[3], "y-", linewidth=1, label="d sigma=%s"%sigma[3])
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(x_data, y_data[2], "r-", linewidth=1, label="sigma=%s"%sigma[2])
plt.plot(x_data, y_d_data[2], "b-", linewidth=1, label="d sigma=%s"%sigma[2])
plt.grid(True)
plt.legend()

plt.show()
