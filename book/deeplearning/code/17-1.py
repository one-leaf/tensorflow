#Gibbs采样
import random
import math
import matplotlib.pyplot as plt

'''
假设（x,y）是一个二元分布，整体服从二项正态分布采样 Norm(m, sum)
且条件概率为
    x = Norm(m1+p(y-m2),sqrt(1-p^2)*m2)
    y = Norm(m2+p(x-m1),sqrt(1-p^2)*m1)
'''

def p_ygivenx(x, m1, m2):
    return (random.normalvariate(m1 + rho * (x - m2), math.sqrt(1 - rho ** 2) * m2 ))

def p_xgiveny(y, m1, m2):
    return (random.normalvariate(m2 + rho * (y - m1), math.sqrt(1 - rho ** 2) * m1 ))

N = 5000
K = 20
x_res = []
y_res = []
m1 = 20
m2 = 10

rho = 0.5
y = m2

for i in range(N):
    for j in range(K):
        x = p_xgiveny(y, m1, m2)
        y = p_ygivenx(x, m1, m2)
        x_res.append(x)
        y_res.append(y)

num_bins = 50
plt.hist(x_res, num_bins, normed=1, facecolor='green', alpha=0.5)
plt.hist(y_res, num_bins, normed=1, facecolor='red', alpha=0.5)
plt.title('Histogram')
plt.show()