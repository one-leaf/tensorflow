# MCMC 采样
import numpy as np
import matplotlib.pyplot as plt
import math

def norm(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 /(2* sigma **2))/(math.sqrt(2*math.pi)*sigma)

def plot_mcmc(mu, sigma):
    cur = np.random.rand()
    states = [cur]
    for i in range(10**5):
        next, u = np.random.rand(2)
        if u < np.min((norm(next, mu, sigma)/norm(cur, mu, sigma), 1)):
            states.append(next)
            cur = next
        else:
            states.append(cur)

    x = np.arange(0, 1, .01)
    plt.figure()
    plt.plot(x, norm(x, mu, sigma), lw=2, label='real dist: a={}, b={}'.format(mu, sigma))
    plt.hist(states, 25, normed=True, label='simu mcmc: a={}, b={}'.format(mu, sigma))
    plt.show()

if __name__ == '__main__':
    plot_mcmc(0.1, 0.1)
    plot_mcmc(0.2, 0.2)
    plot_mcmc(1, 1)