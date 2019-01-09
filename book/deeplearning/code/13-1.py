# 因子分析
# 2018年6-11月东莞市主要经济指标
# 数据来源 http://tjj.dg.gov.cn/website/web2/art_view.jsp?articleId=14062
# 参考: https://kuaibao.qq.com/s/20181021A0VSZ200?refer=cp_1026

import pandas as pd 
import numpy as np

data='''{"columns":["地区生产总值","工业增加值","固定资产投资","零售总额","消费价格指数","公共预算收入","公共预算支出","进出口总额","出口总额","外币存款余额","外币贷款余额","全社会用电量","工业用电"],
     "index":["6月","7月","8月","9月","10月","11月"],
    "data":[[3868.60,330.03,160.15,242.33,101.9,53.97,106.83,1121.5,656.0,13677.63,7560.80,77.15,52.09],
            [3868.60,331.24,147.55,232.13,102.1,61.13,82.75,1239.5,752.3,13736.11,7676.60,85.07,60.20],
            [3868.60,353.73,147.60,239.79,102.7,26.30,36.74,1267.5,768.9,13911.11,7934.77,85.69,58.16],
            [6073.34,358.00,208.89,248.81,102.8,51.33,63.46,1337.0,783.1,14051.37,8026.05,75.08,52.38],
            [6073.34,339.31,152.58,267.09,103,51.37,34.50,1227.35,736.22,14167.01,8082.84,65.98,47.28],
            [6073.34,355.03,194.11,248.61,102.6,59.86,32.74,1229.24,768.35,14287.12,8183.39,64.60,48.63]]}'''

X = pd.read_json(data, orient='split')
print('-'*100, '\n东莞经济指标\n', X)

# 均值规范化
X1 = (X-X.mean())/X.std()
print('-'*100, '\n均值化\n', X1)

# 相关系数矩阵
C = X1.corr()
print('-'*100, '\n相关系数矩阵\n', C)

import numpy.linalg as nlg #导入nlg函数，linalg=linear+algebra

# 计算特征值和特征向量
eig_value, eig_vector = nlg.eig(C)
print('-'*100, '\n特征值\n', eig_value)
print('-'*100, '\n特征向量\n', eig_vector)

# 将列名和特征值创建数据
eig = pd.DataFrame()
eig['names']=X.columns
eig['eig_value']=eig_value
print('-'*100, '\n特征值数据集\n', eig)

# 确定公共因子个数
# 如果解释度达到95%，结束循环，这里得出k=4
for k in range(1,14):
    if eig["eig_value"][:k].sum()/eig["eig_value"].sum()>0.95:
        print('-'*100, '\n公共因子个数\n', k)
        break

col0=np.sqrt(eig_value[0])*eig_vector[:,0] # 因子载荷矩阵第1列
col1=np.sqrt(eig_value[1])*eig_vector[:,1] # 因子载荷矩阵第2列
col2=np.sqrt(eig_value[2])*eig_vector[:,2] # 因子载荷矩阵第3列
col3=np.sqrt(eig_value[3])*eig_vector[:,3] # 因子载荷矩阵第4列
A=pd.DataFrame([col0,col1,col2,col3]).T # 构造因子载荷矩阵A
A.columns=['factor1','factor2','factor3','factor4'] # 因子载荷矩阵A的公共因子
A.index=X.columns
print('-'*100, '\n因子载荷矩阵\n', A)

h=np.zeros(13) # 变量共同度，反映变量对共同因子的依赖程度，越接近1，说明公共因子解释程度越高，因子分析效果越好
D=np.mat(np.eye(13)) # 特殊因子方差，因子的方差贡献度 ，反映公共因子对变量的贡献，衡量公共因子的相对重要性
A=np.mat(A) # 将因子载荷阵A矩阵化

for i in range(13):
    a=A[i,:]*A[i,:].T # A的元的行平方和
    h[i]=a[0,0].real  # 计算变量X共同度,描述全部公共因子F对变量X_i的总方差所做的贡献，及变量X_i方差中能够被全体因子解释的部分
    D[i,i]=1-a[0,0].real # 因为自变量矩阵已经标准化后的方差为1，即Var(X_i)=第i个共同度h_i + 第i个特殊因子方差

print('-'*100, '\n变量共同度\n', h)
print('-'*100, '\n因子方差\n', D)

from numpy import eye, asarray, dot, sum, diag #导入eye,asarray,dot,sum,diag 函数
from numpy.linalg import svd #导入奇异值分解函数

def varimax(Phi, gamma = 1.0, q =20, tol = 1e-6): #定义方差最大旋转函数
    p,k = Phi.shape #给出矩阵Phi的总行数，总列数
    R = eye(k) #给定一个k*k的单位矩阵
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R) #矩阵乘法
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda)))))) #奇异值分解svd
        R = dot(u,vh) #构造正交矩阵R
        d = sum(s) #奇异值求和
        if d_old!=0 and d/d_old: 
            return dot(Phi, R) #返回旋转矩阵Phi*R

rotation_mat=varimax(A) #调用方差最大旋转函数
rotation_mat=pd.DataFrame(rotation_mat) #数据框化
rotation_mat.index=X.columns
print('-'*100, '\n旋转矩阵\n', rotation_mat)

X1=np.mat(X1) #矩阵化处理
factor_score=(X1).dot(A) #计算因子得分
factor_score=pd.DataFrame(factor_score)#数据框化
factor_score.columns=['因子1','因子2','因子3','因子4'] #对因子变量进行命名
factor_score.index=X.index
print('-'*100, '\n因子得分\n', factor_score)