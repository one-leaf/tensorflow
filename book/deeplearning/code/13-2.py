#!/usr/bin/env python

# 样例来源： https://blog.csdn.net/lizhe_dashuju/article/details/50263339
# FastICA from ICA book, table 8.4 

import math
import random
import matplotlib.pyplot as plt
from numpy import *

# 两种信号混合
n_components = 2

def f1(x, period = 4):
    return 0.5*(x-math.floor(x/period)*period)

def create_data():
    # 样本个数
    n = 100
    # 时间轴
    T = [0.1*xi for xi in range(0, n)]
    # 原始分离数据，第一个是sin()，第二个是f1
    S = array([[sin(xi)  for xi in T], [f1(xi) for xi in T]], float32)
    # 将矩阵数据用A混合
    A = array([[0.8, 0.2], [-0.3, -0.7]], float32)
    return T, S, dot(A, S)

# 白化
def whiten(X):
    #zero mean
    X_mean = X.mean(axis=-1)
    X -= X_mean[:, newaxis]
    #whiten
    A = dot(X, X.transpose())
    D , E = linalg.eig(A)
    D2 = linalg.inv(array([[D[0], 0.0], [0.0, D[1]]], float32))
    D2[0,0] = sqrt(D2[0,0]); D2[1,1] = sqrt(D2[1,1])
    V = dot(D2, E.transpose())
    return dot(V, X), V

def _logcosh(x, fun_args=None, alpha = 1):
    gx = tanh(alpha * x, x); g_x = gx ** 2; g_x -= 1.; g_x *= -alpha
    return gx, g_x.mean(axis=-1)

def do_decorrelation(W):
    #black magic
    s, u = linalg.eigh(dot(W, W.T))
    return dot(dot(u * (1. / sqrt(s)), u.T), W)

def do_fastica(X):
    n, m = X.shape; p = float(m); g = _logcosh
    #black magic
    X *= sqrt(X.shape[1])
    #create w
    W = ones((n,n), float32)
    for i in range(n): 
        for j in range(i):
            W[i,j] = random.random()

    #compute W
    maxIter = 200
    for ii in range(maxIter):
        gwtx, g_wtx = g(dot(W, X))
        W1 = do_decorrelation(dot(gwtx, X.T) / p - g_wtx[:, newaxis] * W)
        lim = max( abs(abs(diag(dot(W1, W.T))) - 1) )
        W = W1
        if lim < 0.0001:
            break
    return W

def show_data(T, S, title):
    plt.figure(title)
    plt.ylim(-2,2)
    plt.plot(T, [S[0,i] for i in range(S.shape[1])], marker="*", linewidth=1, markersize=2)
    plt.plot(T, [S[1,i] for i in range(S.shape[1])], marker="o", linewidth=1, markersize=2)
    plt.legend()

def main():
    T, S, D = create_data()
    Dwhiten, K = whiten(D)
    W = do_fastica(Dwhiten)
    #Sr: reconstructed source
    Sr = dot(dot(W, K), D)

    show_data(T, S, "原始信号") # 原始混合前的信号
    show_data(T, D, "混合信号") # 原始混合后的信号
    show_data(T, Sr, "ICA分离信号") # ICA分离后的信号
    plt.show()

if __name__ == "__main__":
    main() 
