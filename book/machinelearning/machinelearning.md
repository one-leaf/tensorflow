对周志华的机器学习（西瓜书）的一些备忘

1. 绪论
 
    - 归纳偏好

        > 任何机器学习都有对某种类型假设的偏好

        > 一般性原则：奥卡姆剃刀，如果有多个假设与观察一致，选择最简单的。

        > 总误差与学习算法无关，称为“没有免费的午餐”定理
        
        > 学习算法的相对优劣需要针对具体的问题，需要归纳偏好与问题相匹配。

    -  多释原则，保留与经验观察一致的所有假设；这个和集成学习的研究相吻合。

2. 模型评估与选择

    - 经验误差与过拟合
    
        > m个样本中有a个分类错误：错误率=a/m、精度=1-a/m
        
        > 误差，训练误差/经验误差，新样本上：泛化误差。
        
        > 过拟合、欠拟合

    - 评估方法
       
       > 留出法 数据集D 分为 训练集S 和 测试集T  S,T 互斥 

       > 交叉验证法 将数据集分为k个大小相同的子集，然后k-1为训练集，1为测试集

       > 自助法 将数据集D随机重复采样m次得到D'，这样会有38%的样本没有被采样，将 D' 作为训练集， D\D' 作为测试集。

    - 性能度量
       
       > 回归，性能度量为均方误差

       > 查准率、查全率、F1

       > TP 真正例 FP 假正例    FN 假反例 TN 真反例

       > 查准率 P = TP / (TP+FP) 查全率 R = TP / (TP+FN)

       > P-R 曲线 差准率为纵轴，查全率为横轴；

       > 如果一个学习器的P-R曲线被另一个包住，则后者优于前者 

       > 平衡点：查准率=查全率

       > F1 度量： F1 = 2 x P x R /(P+R) = 2xTP/(样本总数+TP-TN)

       > Fb = (1+b^2)*P*R/(b^2*P+R) b>0 查全率更重要， b<0 查准率更重要， b=1 ,Fb=F1

       > ROC , 纵轴 TPR = TP/(TP+FN) ，横轴 FPR=FP/(TN+FP)

       > ROC 和 P-R 曲线一样，如果一个学习器的P-R曲线被另一个包住，则后者优于前者

       > AUC ROC曲线下的面积

       > 非均等代价，代价曲线

    - 比较检验

       > 泛化错误率

       > 二项检验

       > 交叉验证t检验

       > McNemar 检验， Friedman 检验

       > 泛化误差可分解为偏差、方差和噪声之和

3. 线性模型
    - 基本形式
        > $$ f(x) = w_1*x_1 + w_2x_2 + ... + w_dx_d + b $$
        > $$ f(x) = w^tx + b $$

    - 线性回归
        > 欧式距离/均方误差 --> 最小二乘法
        
        > $$ (w^*,b^*)=argmin_{(w,b)}\sum^{m}_{i=1}(f(x_i)-y_i)^2 $$ 
        > $$ = argmin_{(w,b)}\sum^{m}_{i=1}(y_i-wx_i-b)^2 $$

        > 对 w 和 b 求导

        > $$ w =\dfrac{\sum^{m}_{i=1}y_{1}\left( x_{i}-\overline {x}\right) }{\sum^{m}_{i=1}x^{2}_{1}-\dfrac {1}{m}\left( \sum^{m}_{i=1}x_{i}\right) ^{2}} $$ 
    
        > $$ b=\dfrac{1}{m}\sum^m_{i=1}(y_i-wx_i) $$

        > $$ \overline x = \dfrac1m\sum^m_{i=1}x_i $$
        
        > 多元线性回归

        > $$ f(x_i) = w^Tx_i+b \rightarrow f(x_i) \simeq y_i $$

        > 对数线性回归

        > $$ lny=w^tx+b $$
        > $$ y = e^{w^tx + b} $$

        > 对数几率回归 

        > sigmoid 函数

        > $$ y = \dfrac {1}{1+e^{-z}} $$

        > $$ y = \dfrac {1}{1+e^{-(w^tx+b)}} $$

        > $$ ln\dfrac {y}{1-y} = w^tx+b $$

        > y 为 x 的正例可能性 1-y 为 x 的反例可能性

        > 几率

        > $$ \dfrac {y}{1-y} $$

        > 对数几率

        > $$ ln\dfrac {y}{1-y} $$

        > 极大似然法估算wb，采用梯度下降或牛顿法求最优解

    - 线性判别分析

        > 给定训练样本，将样本投射到一条直线上，使得同类样本的投影点尽可能的接近，异类的投影点尽可能远离

    - 多分类学习

        > 一般用拆解法，将多分类拆解为多个二分类，分为正类和其他类

        > OvO OvR MvM ECOC

    - 类别不平衡问题
        
        > 欠采样，过采样

        > 阈值移动

        > $$ \dfrac {y'}{1-y'} = \dfrac {y}{1-y} * \dfrac {m^-}{m^+} $$

    - 阅读材料

        > 稀疏表示

        > 多标记学习


 4. 决策树       

    - 划分选择
        > 信息增益

        > 信息熵 样本D中第k类样本所占的比例为 p<sub>k</sub> (k=1,2,...,|y|), 则D的信息熵定义为：

        > $$Ent(D) = - \sum ^{|y|}_{k=1} p_klog_2p_k $$

        > Ent(D) 的值越少，D的纯度越高

        > 西瓜数据集 17 个样本，2个分类（好瓜、怀瓜），正例8个，反例9个，求 信息熵

        > $$ Ent(D) = - \sum ^2_1 p_klog_2p_k = -(\dfrac {8}{17}log_2\dfrac {8}{17} + \dfrac {9}{17}log_2\dfrac {9}{17}) = 0.998 $$

        > 信息增益越大，则使用属性a来划分所获得的纯度提升越大。可以用信息增益来进行决策树的划分属性

        > 增益率对可取值少的属性有偏好，因此应该先找到信息增益大于平均值的划分属性，然后再选增益率最高的

        > 基尼指数越小，数据的纯度越高

    - 剪枝处理

        > 防止过拟合

    - 连续和缺失值

        > 连续值直接用二分法处理，计算信息增益

        > 缺失值，先默认权重为1，然后缺失的数据直接并入各子节点，同时修改各子节点的权重

    - 多变量决策树

        >  直接用线性分类器

5. 神经网络

    - 神经元模型

        > M-P 神经元模型

        > $$ y = f(\sum_{i=1}^{n}w_ix_i-\theta)$$

        > 激活函数，有阶跃函数 Sigmoid

        > $$ sgn(x)=\begin{cases}1,x\geq 0\\ 0,x <0\end{cases} $$

        > $$ sigmoid(x)=\dfrac 1{1+e^{-e}} $$

    - 感知机与多层网络

        > 感知机由两层神经元组成,可以方便的进行逻辑运算（假定激活函数为阶跃函数）

        > 与： x1 ^ x2 , 令 w1 = w2 = 1, b = 2, 则 y = f(1*x1 + 1*x2 -2) 仅 x1 = x2 = 1 时,y = 1

        > 或： x1 | x2 , 令 w1 = w2 = 1, b = 0.5, 则 y = f(1*x1 + 1*x2 - 0.5) 当 x1 或 x2 = 1 时,y = 1

        > 非： -x1 ， 令 w1 = 0.6, w2 = 0, b = -0.5, 则 y = f(-0.6*x1 + 0*x2 + 0.5) 当 x1 = 1，y=0；x1 =0，y=1 

        > 学习率

        > 简单的两层无法解决非线性的异或问题，引入多层前馈神经网络，前馈避免网络中出现环路或回路

    - 误差逆传播算法

        > BP算法，基于梯度下降算法

        > 训练前期SGD占优，后期标准梯度下降占优

        > 只需要一个包含足够多的神经元的隐含层，可以已任意精度逼近任意复杂的连续函数

    - 全局最小与局部最小

        > 多组不同参数的初始化多个神经网络

        > 模拟退火，每一步有一定概率都接收一个更差的结果，该概率也逐步下降

        > 随机梯度下降

    - 其它常见神经网络

        > RBF ART SOM 级联相关网络 Elman网络 Boltzmann机    

    - 深度学习

        > 逐层训练 权重共享

6. 支持向量机

    - 间隔与支持向量

        > 划分超平面方程

        > $$w^Tx+b=0$$ 

        > w=(w1;w2...;wd) 为法向量，决定超平面的方向；b为位移项，决定超平面到原点之间的距离

        > 样本空间任意点x到超平面(w,b)的距离为

        > $$ r = \dfrac {|w^Tx+b|}{||w||} $$

        > 求最大间隔

        > $$ r = \dfrac {2}{||w||} $$

        > $$ \min _{w,b}\dfrac {1}{2}\left\| w\right\| ^{2} $$ 

    - 对偶问题

        > SMO 

    - 核函数

        > 将无法线性可分的样本，映射到更高维度，达到线性可分

        > 线性核、多项式核、高斯核、拉普拉斯核、Sigmoid核

    - 软间隔与正则化

        > 为了防止过拟合，允许向量机在一些样本上出错

        > 软间隔常用替代损失函数：hinge 损失 max(0,1-z)、指数损失 exp(-z)、对数损失 log(1+exp(-z))

    - 支持向量回归

        > 正常的回归是f(x)与y的差，向量回归是 f(x) 与 y 的差的绝对值大于 e 才计算损失。

        > f(x) +/- e 间隔带

        > 拉格朗日函数

        > $$ f(x)=\sum _{i=1}^{m}(\hat {a_{i}}-a_i)k(x,x_i)+b $$

        > $$ k(x,x_i) = \phi (x_i)^T \phi(x_j) $$ 
    
    - 核方法

        > 表示定理

        > 引入核函数，将线性机器学习拓展为非线性机器学习

7. 贝叶斯分类器

    - 贝叶斯决策论
        > 最小化决策风险

    - 极大似然估计
        > 根据数据采用来估算概率分布

    - 朴素贝叶斯分类器
        > 为了克服有限的样本问题，按每个属性估计条件概率

    - 半朴素贝叶斯分类器
        > 对属性引入超父
    
    - 贝叶斯网
        > 道德图

        > 学习，最小描述长度

        > 推断，马尔可夫链

        > EM算法

    - 阅读材料
        > 结构学习 和 参数学习


8. 集成学习

    - 个体与集成
        > 多个个体学习器，结合到一起输出

        > 基学习器的误差相互独立

    - Boosting
        > 基学习器，对特定的数据分布进行学习

        > 降低偏差

    - Bagging 与随机森林

        > 自助采样法（63.2%）训练基学习器

        > 降低方差

        > 随机森林引入属性扰动

    - 结合策略
        > 提高泛化性能

        > 避免局部最小点

        > 平均法、投票法、学习法（初级学习器 --> 次级学习器/元学习器）
    
    - 多样性
        > 误差-分歧分解

        > 多样性度量

        > 多样性增强（样本扰动、属性扰动、输出表示扰动、算法参数扰动）

9. 聚类

    - 聚类任务
        > 无监督学习分类

    - 性能度量
        > DBI 和 DI 指数

    - 距离计算
        > VDM 和 加权闵可夫斯基距离

    - 原型聚类
        > k均值算法

        > 学习向量量化

        > 高斯混合聚类
    
    - 密度聚类
        > DBSCAN 算法
    
    - 层次聚类
        > AGNES 算法

    - 阅读材料
        > 距离计算方法，闵可夫斯基距离、内积距离、余弦距离

10. 降维与度量学习
    - k 近邻学习
        > 懒惰学习

    - 低维嵌入
        > 维度灾难

        > 线性变换
    
    - 主成分分析（PCA）
        > 降噪用处

    - 核化线性降维
        > 非线性降维

        > 先通过核函数将特征映射到高维，再用PCA降维

    - 流形学习
        > 等度量映射
            > K近邻加回归

        > 局部线性嵌入
    
    - 度量学习
        > 对度量矩阵进行学习

    - 阅读材料
        > 无监督降维,PCA

        > 监督降维,LDA

11. 特征选择与稀疏学习

    - 子集搜索与评价

    - 过滤式选择，Relief

    - 包裹式选择, LVW    

    - 嵌入式选择与L1正则化

    - 稀疏表示与字典学习

    - 压缩感知
        > 基寻踪去噪

        > 矩阵补全
    
    - 阅读材料
        > LARS

12. 计算学习理论
    - 基础知识
        > Jensen 不等式
            > $$ f(\dfrac {x_1+x_2+...+x_n} {n}) \leq \dfrac {f(x_1)+f(x_2)+...+f(x_n)}{n} $$

        > Hoeffding 不等式

        > McDiarmid 不等式

    - PAC 学习
    
    - 有限假设空间
        > 可分和不可分情形

    - VC 维

    - Rademacher 复杂度

    - 稳定性

    - 阅读材料

13. 半监督学习

    - 未标记样本

        > 纯半监督学习，未标记数据为待预测数据

        > 直推学习，未标记数据不为待预测数据

    - 生成式方法

    - 半监督 SVM

    - 图半监督学习

    - 基于分歧的方法

    - 半监督聚类

    - 阅读材料

14. 概率图模型

    - 隐马尔科夫模型

    - 马尔科夫随机场

    - 条件随机场

    - 学习和推断
        > 变量消去

        > 信念传播

    - 近视推断

    - 话题模型

    - 阅读材料

15. 规则学习

    - 基本概念
        > $$ \oplus\leftarrow f_{1}\wedge f_{2}...\wedge f_{L} $$

    - 序贯覆盖

    - 剪枝优化

    - 一阶规则学习

    - 归纳逻辑程序设计
        > 最小一般泛化

        > 逆归结

    - 阅读材料

16. 强化学习

    - 任务和奖赏
        > T步累积奖励

        > r折扣累积奖励

        > 强化学习可以看做是具有延迟标记信息的监督学习问题

    - K-摇臂赌博机
        > 探索与利用窘境

        > E-贪心

            平均奖赏

         $$ Q_n(k) = \dfrac {1}{n}((n-1) \times Q_{n-1}(k)+v_n) $$

            softmax

        > 有模型学习
            > 策略评估
            > 策略改进
            > 策略迭代与值迭代
        
        > 免模型学习
            > 蒙特卡罗强化学习
            > 时序差分学习（Q-learning）
            > 值函数近似

        > 模仿学习
            > 直接模仿学习
            > 逆强化学习

        > 阅读材料

             