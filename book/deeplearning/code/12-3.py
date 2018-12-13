# 重要性采样
import numpy as np
import matplotlib.pyplot as plt

# 随机采样
def qsample():
    return np.random.rand()*4.

# 原函数分布
def p(x):
    return 0.3*np.exp(-(x-0.3)**2) + 0.7* np.exp(-(x-2.)**2/0.3) 

# 提议分布
def q(x):
    return 4.0

# 计算权重
def importance(nsamples):
    samples = np.zeros(nsamples,dtype=float)
    w = np.zeros(nsamples,dtype=float)
    
    for i in range(nsamples):
        samples[i] = qsample()
        w[i] = p(samples[i])/q(samples[i])
                
    return samples, w

# 按权重得到采样序号
def sample_discrete(vec):
    u = np.random.rand()
    start = 0
    for i, num in enumerate(vec):      
        if u > start:
            start += num
        else:
            return i-1
    return i

# 重要性采样
def importance_sampling(nsamples):
    samples, w = importance(nsamples)
    print("samples", samples)
    print("w", w)
    final_samples = np.zeros(nsamples,dtype=float)
    w = w / w.sum()
    for j in range(nsamples):
        final_samples[j] = samples[sample_discrete(w)]
    return final_samples

x = np.arange(0,4,0.01)
realdata = p(x) 
plt.plot(x,realdata,'g',lw=6)

# samples,w = importance(5000)
# plt.hist(samples, normed=1, fc='r')

final_samples = importance_sampling(5000)
plt.hist(final_samples, normed=1, fc='c')
plt.show()