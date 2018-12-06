# 绘制Beta曲线
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np

params = [0.5, 1, 2, 3]
x = np.linspace(0, 1, 100)
f, ax = plt.subplots(len(params), len(params), sharex=True, sharey=True)
for i in range(4):
    for j in range(4):
        alpha = params[i]
        beta = params[j]
        pdf = ss.beta(alpha, beta).pdf(x)
        ax[i, j].plot(x, pdf)
        ax[i, j].plot(0, 0, label='alpha=%s\nbeta=%s'%(alpha, beta), alpha=0)
        plt.setp(ax[i, j], xticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0,2,4,6,8,10])
        ax[i, j].legend(fontsize=10)
ax[3, 0].set_xlabel('theta', fontsize=16)
ax[0, 0].set_ylabel('pdf(theta)', fontsize=16)
plt.suptitle('Beta PDF', fontsize=16)
plt.tight_layout()

plt.figure()
pdf = ss.beta(10, 1).pdf(x)
plt.plot(x, pdf, 'g-', label='alpha=%s\nbeta=%s'%(10, 1))
pdf = ss.beta(10, 2).pdf(x)
plt.plot(x, pdf, 'r-', label='alpha=%s\nbeta=%s'%(10, 2))
plt.legend()

plt.show()
