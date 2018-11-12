# 高斯混合聚类
import random
import matplotlib.pyplot as plt
import numpy as np

# 定义多少个組件
K = 4
# 数据集维度，目前2维
D = 2
# 样本个数
N = 200

# 按K产生随机数据
def loadDataSet(count=200):
    center_point = np.random.random((K, 2))
    data = np.zeros((count, D))
    for i in range(count):
        data[i, :] = center_point[i % K, :]+np.random.normal(0, 0.05, (2))
    return data

# 初始化参数
def initParams(K, D):
    # 每个簇的中心值：[K D]
    aves = np.random.rand(K, D)
    # 每个簇的偏差 [D D K]
    sigmas = np.zeros((D, D, K))

    # [D D] 必须是对称矩阵
    # 1 0
    # 0 1
    sig = np.eye(D)
    for k in np.arange(0, K):
        sigmas[:, :, k] = sig

    # 每个簇的影响系数：[1 K], 初始化為平均值
    pPis = np.ones((1, K)) / K 
    return aves, sigmas, pPis

# 样本点对簇的贡献系数
# pPi : [1 K]
# px: [N K]
# return value: [N K]
def fgamma(px, pPi):
    z = pPi * px
    s = np.sum(z,axis=1)
    s = s[:, np.newaxis]
    return z/s

# 每个簇中的样本点的贡献系数之和
# gam: [N K]
# return value: [1 K]
def fNk(gam):
    nk = np.sum(gam, axis=0)
    return nk[np.newaxis, :]

# 每个簇的均值
# Nk: [1 K]
# gam: [N K]
# x : [N D]
# return value: [K D]
def faverage(aves, Nk, gam, x):
    for k in np.arange(0, K):
        sumd = np.sum((gam[:, k].reshape(N, 1)) * x, axis=0)
        aves[k, :] = sumd.reshape(1, D)/Nk[:, k]

# 每个簇的方差
# Nk: [1 K]
# gam: [N K]
# x : [N D]
# aves: [K D]
# return value: [D D K]
def fsigma(sigmas, Nk, gam, x, aves):
    for k in np.arange(0, K):
        # shift: [N DA]
        shift = x - aves[k, :]
        # shift_gam: [N D]
        shift_gam = gam[:, k].reshape(N, 1)*shift
        # shift2 : [D D]
        shift2 = shift_gam.T.dot(shift)
        sigmas[:, :, k] = shift2/Nk[:, k]
    return sigmas

# D-维高斯分布的概率密度
# x ： [N D]
# aves : [K D]
# sigmas: [D D K]
# return value: [N K]
def fpx(x, aves, sigmas):
    Px = np.zeros((N, K))
    # coef1 : [1 1]
    coef1 = np.power((2*np.pi), (D/2.0))
    for k in np.arange(0, K):
        # coef2 : [1 1]
        coef2 = np.power((np.linalg.det(sigmas[:, :, k])), 0.5)
        coef3 = 1/(coef1 * coef2)
        # shift: [N D]
        shift = x - aves[k, :]
        # sigmaInv: [D D]
        sigmaInv = np.linalg.inv(sigmas[:, :, k])
        epow = -0.5*(shift.dot(sigmaInv)*shift)
        # epowsum : N
        epowsum = np.sum(epow, axis=1)
        Px[:, k] = coef3 * np.exp(epowsum)
    return Px


# 迭代求解的停止策略
# px: [N K]
# pPi: [1 K]
# Loss function [1 1]
def fL(px, pPi):
    # sub: [N 1]
    sub = np.sum(pPi*px, axis=1)
    logsub = np.log(sub)
    curL = np.sum(logsub)
    return curL

# stop iterator strategy
def stop_iter(threshold, preL, curL):
    return np.abs(curL-preL) < threshold

# GMM
# return value: [N K]
def GMM(x, K):
    # loss value initilize
    preL = -np.inf
    # aves 每个簇的中心值：[K D]
    # sigmas 每个簇的偏差 [D D K]
    # pPi 每个簇的影响系数：[1 K]
    aves, sigmas, pPi = initParams(K, D)
    while True:
        # px: 每个数据所属簇的概率 [N K]
        px = fpx(x, aves, sigmas)
        # print(px)
        # 贡献系数 [N K]
        gam = fgamma(px, pPi)
        # 每个簇中的样本点的贡献系数之和 [1 K]
        Nk = fNk(gam)
        pPi = Nk/N
        # 每个簇的均值 [K D]
        faverage(aves, Nk, gam, x)
        # 每个簇的方差 [D D K]
        fsigma(sigmas, Nk, gam, x, aves)
        # loss function
        curL = fL(px, pPi)
        # 迭代求解的停止策略
        if stop_iter(1e-10, preL, curL):
            break
        preL = curL
    return px, aves, sigmas

# 返回聚类的结果：N
def classifior(px):
    rslt = []
    for row in px:
        rslt.append(np.where(row == np.max(row)))
    return np.array(rslt).reshape(-1)

def main():
    x = loadDataSet(N)

    # 二维特征的GMM聚类模拟
    px, aves, sigmas = GMM(x, K)
    mylabel = classifior(px)

    # 绘制
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=mylabel)
    plt.scatter(aves[:, 0], aves[:, 1], c="r", marker="*", s=100)
    plt.show()


if __name__ == '__main__':
    main()
