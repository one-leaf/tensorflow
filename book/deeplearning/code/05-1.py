# pca 分解
import numpy as np

# X 是输入的矩阵，k是分类个数
def pca(X,k):
    # 按列求均值
    n_samples, n_features = X.shape
    mean = np.mean(X, axis=0)

    # 归一化
    norm_X=X-mean

    # 转为方阵
    scatter_matrix=np.dot(norm_X.T,norm_X)
    
    #计算特征值和特征向量
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)

    # 对特征向量按特征值从高往低排序
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
    eig_pairs.sort(reverse=True)

    # 选择前面k个特征向量
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
  
    # 映射数据
    data=np.dot(norm_X, feature.T)
    return data

if __name__ == '__main__':
    x=np.array([[1,2],[5,4],[3,6],[9,8]])
    print(pca(x, 1))
    