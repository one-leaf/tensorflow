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

# 这里直接用平均值，使用标准softmax会指数太大
def softmax(z):
    s = np.sum(z,axis=1)
    s = s[:, np.newaxis]
    return z/s

# 初始化参数
def initParams(K, D):
    # 每个簇的中心值：[K D]
    aves = np.random.rand(K, D)
    # 每个簇的协方差矩阵 [D D K]
    sigmas = np.zeros((D, D, K))

    # [D D] 必须是对称矩阵
    sig = np.eye(D)
    for k in np.arange(0, K):
        sigmas[:, :, k] = sig

    # 每个簇的影响系数：[1 K], 初始化為平均值
    pPis = np.ones((1, K)) / K 
    return aves, sigmas, pPis

# 样本点对簇的贡献系数
# pPi : [1 K] 影响系数
# px: [N K]   概率密度
# return value: [N K]
def fgamma(px, pPi):
    z = pPi * px
    return softmax(z)

# 每个簇中的样本点的贡献系数之和
# gam: [N K]    样本簇概率密度
# return value: [1 K]
def fNk(gam):
    nk = np.sum(gam, axis=0)
    return nk[np.newaxis, :]

# 每个簇的均值
# aves: [K D]   每个簇的中心点
# Nk: [1 K]     影响系数
# gam: [N K]    贡献系数
# x : [N D]     样本
# return value: [K D]
def faverage(aves, Nk, gam, x):
    for k in np.arange(0, K):
        sumd = np.sum((gam[:, k].reshape(N, 1)) * x, axis=0)
        aves[k, :] = sumd.reshape(1, D)/Nk[:, k]

# 每个簇的方差
# Nk: [1 K]     影响系数
# gam: [N K]    贡献系数
# x : [N D]     样本
# aves: [K D]   每个簇的中心点
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

# D-维的概率
# x ： [N D]            样本
# aves : [K D]          每个簇的中心点
# sigmas: [D D K]       每个簇的协方差矩阵
# return value: [N K]
def fpx(x, aves, sigmas):
    Px = np.zeros((N, K))
    # coef1 : [1 1]
    coef1 = np.power((2*np.pi), (D/2.0))
    for k in np.arange(0, K):
        # coef2 : [1 1]
        # 计算行列式的开方
        coef2 = np.power(np.linalg.det(sigmas[:, :, k]), 0.5) 
        coef3 = 1/(coef1 * coef2)
        # shift: [N D]
        shift = x - aves[k, :]
        # sigmaInv: [D D] 加了一个随机数防止无解
        sigmaInv = np.linalg.inv(sigmas[:, :, k] + 1e-8*np.random.rand(D, D))
        epow = -0.5*(shift.dot(sigmaInv)*shift)
        # epowsum : N
        epowsum = np.sum(epow, axis=1)
        Px[:, k] = coef3 * np.exp(epowsum)
    return Px

# 迭代求解的停止策略
# px: [N K]         概率密度
# pPi: [1 K]        影响系数
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
    # aves      每个簇的中心值：[K D]
    # sigmas    每个簇的协方差矩阵： [D D K]
    # pPi       每个簇的影响系数：[1 K]
    # 初始化高斯混合成分参数
    aves, sigmas, pPi = initParams(K, D)
    for _ in range(100):
        # px: 每个数据所属簇的概率 [N K]
        # 计算混合成分生成的后验概率
        px = fpx(x, aves, sigmas)

        # 贡献系数 [N K]
        gam = fgamma(px, pPi)
        # 每个簇中的样本点的贡献系数之和 [1 K]
        Nk = fNk(gam)
        pPi = Nk/N
        # 计算每个簇的均值向量 [K D]
        faverage(aves, Nk, gam, x)
        # 计算每个簇的新协方差矩阵 [D D K]
        fsigma(sigmas, Nk, gam, x, aves)
        # loss function
        curL = fL(px, pPi)
        # 迭代求解的停止策略
        if stop_iter(1e-16, preL, curL):
            break
        preL = curL
    return px, aves, sigmas

# 返回聚类的结果：N
def classifior(px):
    rslt = []
    for row in px:
        rslt.append(np.where(row == np.max(row)))
    return np.array(rslt).reshape(-1)

# 计算聚类得分，越大越好
def calinski_harabasz_score(x, labels):
    extra_disp, intra_disp = 0., 0.
    mean = np.mean(x, axis=0)
    for k in range(K):
        cluster_k = x[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)
    return (1. if intra_disp == 0. else
            extra_disp * (N - K) / (intra_disp * (K - 1.)))

def main():
    x = loadDataSet(N)

    # 二维特征的GMM聚类模拟
    score = 0
    for i in range(10):
        _px, _aves, _sigmas = GMM(x, K)
        _mylabel = classifior(_px)
        _score = calinski_harabasz_score(x, _mylabel)

        if score < _score:
            aves = _aves
            mylabel = _mylabel
            score = _score

    # 绘制
    plt.scatter(x[:, 0], x[:, 1], marker='o', c=mylabel)
    plt.scatter(aves[:, 0], aves[:, 1], c="r", marker="*", s=100)
    plt.show()


if __name__ == '__main__':
    main()
