DEEP LEARNING 学习笔记

2. 线性代数

    - 标量、向量、矩阵、张量
    
        > 标量(scalar)： 单独的数

        > 向量(vector)： 一列数，可以看做空间中的点

        > 矩阵(matrix)： 二维数组

        > 张量(tensor)： 多维数组

        转置(transpose)，将矩阵按对角线为轴镜像: 

        $$ (A^T)_{i,j} = A_{j,i} $$

        例子：

        $$ A = \begin{bmatrix} A_{1,1} & A_{1,2} \\ A_{2,1} & A_{2,2} \\ A_{3,1} & A_{3,2} \end{bmatrix}\Rightarrow A^T = \begin{bmatrix} A_{1,1} & A_{2,1} & A_{3,1} \\ A_{1,2} & A_{2,2} & A_{3,2} \end{bmatrix} $$

    - 矩阵和向量相乘

        > 如果A的形状是(m,n), B的形状是(n,p), C的形状是(m,p)，则可以书写为 C = AB

        定义：

        $$ C_{i,j} = \sum _{k}A_{i,k}B_{k,j} $$

        注意不同于点积 A * B

        一些性质：
        
        $$ A(B+C) = AB + AC $$
        
        $$ A(BC) = (AB)C $$

        $$ x^Ty=y^Tx $$

        $$ (AB)^T=B^TA^T $$

        $$ x^Ty=(x^Ty)^T=y^Tx

    - 单位矩阵和逆矩阵

        任何向量与单位矩阵相乘都不会改变

        $$ \begin{bmatrix} 1&0&0 \\ 0&1&0 \\ 0&0&1 \end{bmatrix} = I_3 $$

        I 就是单位矩阵

        A的逆矩阵为 $A^{-1}$ 

        $$ A^{-1}A=I_n $$

    - 线性相关和生成子空间

        线性方程组

        $$ Ax = b $$

        A 为已知矩阵，b为已知向量，求解向量x

        展开：

        $$ A_{1,1}x_1+A_{1,2}x_2+...+A_{1,n}x_n=b_1 $$
        $$ A_{2,1}x_1+A_{2,2}x_2+...+A_{2,n}x_n=b_2 $$
        $$ ... $$
        $$ A_{m,1}x_1+A_{m,2}x_2+...+A_{m,n}x_n=b_m $$

        如果逆矩阵 $A^{-1}$ 存在，则上面有唯一解，反之则存在无解或存在无限多个解

        方程的解可以看做A的列向量从原点出发的不同方向，确定有多少种办法可以达到向量b；x的每个元素为我们应该沿着这个方向走的距离。

        $$ A_x=\sum _{i} x_iA_{:,i} $$

        这种操作就称为线性组合

        一组向量的生成子空间是原始向量线性组合后所能抵达的点的集合

        确定 Ax=b 是否有解，相当于确定向量b是否在A列向量的生成子空间内。这个子空间称为A的列空间或A的值域。

        向量中的冗余称为线性相关，如果一组向量中的任意一个向量都不能表示成其他向量的线性组合，那么称为线性无关。

        列线性相关的矩阵被称为奇异的。如果矩阵A是一个方阵并且不是一个奇异的，可以用矩阵逆求解。
    
    - 行列式

        矩阵的行列式$|A|$是一个可以从方形矩阵（方阵）计算出的特别的数。用于解线性方程或找逆矩阵。

        2x2矩阵：

        $$A=\begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix}$$
        $$|A|=x_{11}*x_{22}-x_{12}*x_{21}$$

        3x3 矩阵：

        $$A=\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix}$$
        $$|A|=a*\begin{bmatrix} e&f \\ h&i \end{bmatrix}-b*\begin{bmatrix} d&f \\ g&i \end{bmatrix}+c*\begin{bmatrix} d&e \\ g&h \end{bmatrix}$$
        $$|A|=a(ei-fh)-b(di-fg)+c(dh-eg)$$

        4x4 和更大的矩阵：
        
        $$A=\begin{bmatrix} a&b&c&d \\ e&f&g&h \\ i&j&k&l \\ m&n&o&p \end{bmatrix}$$

        $$
        |A|=a*\begin{bmatrix} &f&g&h \\ &j&k&l \\ &n&o&p \end{bmatrix}
        -b*\begin{bmatrix} e&&g&h \\ i&&k&l \\ m&&o&p \end{bmatrix}
        +c*\begin{bmatrix} e&f&&h \\ i&j&&l \\ m&n&&p \end{bmatrix}
        -d*\begin{bmatrix} e&f&g& \\ i&j&k& \\ m&n&o& \end{bmatrix}
        $$

        注意 +-+- 的规律，(+a... -b... +c... -d...) 依次类推。

        这个称为拉普拉斯展开。

    - 范数

        范数是满足下列性质的函数:

        $$ f(x)=0 \Rightarrow x=0 $$
        $$ f(x+y) \le f(x) + f(y) $$
        $$ \forall \alpha\in \mathbb{R}, f(\alpha x)=|\alpha|f(x)  $$

        向量的范数：

        - 0 范数：向量中非零元素的个数
        - 1 范数：绝对值之和

        $$ ||x||_1=\sum _i|x_i| $$

        - 2 范数：通常意义上的模，各元素的平方和再开方

        $$ ||x||_2=\sqrt {\sum _ix_i^2} $$

        - 无穷范数或最大范数：就是取向量的最大值

        $$ ||x||_\infty =\max_j |x_i| $$
        $$ ||x||_{-\infty} =\min_j |x_i| $$

        衡量矩阵的大小为范数（norm）

        - p 范数：向量绝对值的p次方和的1/p次幂

        $$ ||x||_p = (\sum _i|x_i|^p)^{1/p} $$


        当p=2, p 范数等于L2范数，称为欧几里得范数，表示从原点出发到向量x确定的点的欧几里得距离

        L2范数省略开平方后称为平方L2范数。

        平方L2范数在原点附近的增长很缓慢，所有某些场合，采用L1范数，简化为：

        $$ ||x||_1=\sum _i|x_i| $$

        L1范数有时候可以作为统计非0元素数目的替代。

        矩阵的范数：

        - 1 范数，列和范数，所有矩阵列向量之和的最大值

        $$ ||A||_1 = max_j \sum _{i=1}^m|a_{i,j}| $$

        - 2 范数，谱范数，即$A^TA$矩阵的最大特征值的开平方。

        $$ ||A||_2 = \sqrt {\lambda_1} $$

        $\lambda_1$为$A^TA$的最大特征值。

        - 无穷范数或最大范数，行和范数，即所有矩阵行向量绝对值之和的最大值。

        $$ ||A||_F = \max_i\sum _{j=1}^n|a_{i,j}|$$

        - 范数也可以用来衡量矩阵的大小，称为Frobenius范数，简称F范数，就是矩阵的每个元素的平方和的开方。

        $$ ||A||_F = \sqrt {\sum _{i,j}A_{i,j}^{2}} $$

        两个向量的点积可以用范数来表示:

        $$ x^Ty=||x||_2||y||_2cos\theta $$

        $\theta$ 表示 x 和 y 之间的夹角 

        例子，求 矩阵$A =\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$ 的L2范数

        矩阵范数定义

        $$ ||A||^2_2= \sqrt {\lambda_1} $$
        
        $\lambda_1$为矩阵A的最大特征值

        $$ \lambda = A^T A=\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}*\begin{bmatrix} 1 & 4 \\2 & 5 \\ 3 & 6 \end{bmatrix}$$
        $$=\begin{bmatrix} 1*1+2*2+3*3 & 1*4+2*5+3*6 \\ 4*1+5*2+6*3 & 4*4+5*5+6*6 \end{bmatrix}$$
        $$=\begin{bmatrix} 14 & 32 \\ 32 & 77 \end{bmatrix}$$
        $$|\lambda E-A|=\begin{bmatrix} \lambda -14 & -32 \\ -32 & \lambda -77 \end{bmatrix}=0$$
        $$\Rightarrow(\lambda-14)*(\lambda-77)-(-32*-32)=0$$
        $$\Rightarrow\lambda^2-91\lambda+54=0$$
        $$\Rightarrow{(\lambda-{\dfrac {91}{2}})} ^2-\dfrac {91^2}{4}+54=0$$
        $$\lambda= \pm\sqrt{\dfrac {91^2}{4}-54}+\dfrac{91}{2}$$
        $$\lambda=[90.4\ 0.6]$$
        $$||A||^2_2=\sqrt{\max_i\lambda}=\sqrt {90.4}=9.5079$$

    - 特殊类型的矩阵和向量

        对角矩阵：只有主对角线上含有非零元素，其余都是零。

        对称矩阵：矩阵转置后和原矩阵相等。 $A=A^T$

        单位向量：具有单位范数的向量 $||x||_2=1$

        正交矩阵：行向量和列向量是分别标准正交的方阵， $A^TA=AA^T=I$

    - 特征分解

        将矩阵分解为一组特征向量和特征值。

        方阵A的特征向量是指与A相乘后相当于对该向量进行缩放的非零向量v

        $$ Av=\lambda v $$

        $\lambda$ 就是这个特征向量对应的特征值

        例子：求$A=\begin{bmatrix} 1& 1 & -1\\1 &-2&2 \\-3&1&3 \end {bmatrix}$ 的特征值

        1. 根据特征多项式得
        
        $$|\lambda E-A|=\begin{bmatrix} \lambda-1& -1 & 1\\-1 &\lambda+2&-2 \\3&-1&\lambda-3 \end {bmatrix}=0$$
        
        2. 第1行减去第三行，得到一个0
        
        $$=\begin{bmatrix} \lambda-4& 0 & 4-\lambda\\-1 &\lambda+2&-2 \\3&-1&\lambda-3 \end {bmatrix}$$
        
        3. 第3列加上第一列，得到第二个0
        
        $$=\begin{bmatrix} \lambda-4& 0 & 0\\-1 &\lambda+2&-3 \\3&-1&\lambda \end {bmatrix}$$

        4. 展开多项式

        $$=(\lambda-4)\begin{bmatrix} \lambda+2&-3 \\-1&\lambda \end {bmatrix} - 0*... + 0*...$$
        $$=(\lambda-4)((\lambda+2)(\lambda)-(-1*-3))$$
        $$=(\lambda-4)(\lambda^2+2\lambda-3)$$
        $$=(\lambda-4)(\lambda-1)(\lambda+3)=0$$
        
        5. 解得
        
        $$\lambda=[4,1,-3]$$

        矩阵A有n个线性无关的特征向量 {$v^{(1)},...,v^{(1)}$,对应的特征值{$\lambda^{(1)},...,\lambda^{(n)}$}。将特征向量连接成一个矩阵，每一列是一个特征向量: $V=[v^{(1)},...,v^{(n)}]$，类似特征值也可以接连成一个向量，因此A的特征分解可以记为：

        $$A=Vdiag(\lambda)V^{(-1)}$$

    - 奇异值分解

        将矩阵分解为奇异向量和奇异值。非方阵只能用奇异值分解。

        $$A=UDV^T$$

        假设 A:(m,n) 则 U:(m,m) D:(m,n) V:(n,n), 其中U和V都为正交矩阵，D为对角矩阵（不一定是方阵）

        对角矩阵D的对角线上的元素为 A 的奇异值，U的列向量为左奇异向量，V的列向量为右奇异向量。

    - 伪逆

        解非方矩阵

        $$ A^+ = VD^+U^T $$ 

        U、D、V是A奇异值分解后得到的矩阵

    - 迹运算

        迹运算返回的是矩阵对角元素的和：

        $$ Tr(A) = \sum _iA_{i,i} $$

        相当于另外一种范式

    - 行列式

        行列式，det(A) ,是将方阵A映射到实数的函数，行列式等于矩阵特征值的乘积。

    - 主成分分析（PCA）

        简单的机器学习算法，通过压缩信息，编码和解码

3. 概率和信息论

    - 概率

        概率就是信任度，和频率有关的，例如骰子，称为频率派概率；医生诊断流感40%的可能性，涉及到确定性水平，为贝叶斯概率。

    - 随机变量

        可以为离散或连续，随机变量是对可能的状态的描述，必需伴随一个概率分布来指定状态的可能性。

    - 概率分布

        用来描述随机变量的在每一个可能取得的状态的可能性大小

        - 离散型变量和概率质量函数

            通常每一个随机变量都有一个不同的概率质量函数 P(x)。概率质量函数将随机变量能够取得的每个状态映射到随机变量取得该状态的概率。也可以同时作用于多个随机变量，这个称为联合概率 P(x,y)。

        - 连续性变量和概率密度函数

            概率密度函数 p(x) 没有直接对特定状态给出概率，他给出的是落在面积无限小的区域内的概率。可以对概率密度函数求积分来获得点集的真实概率质量。

    - 边缘概率

        已知一组变量的联合概率，求其中一个子集的概率分布，该概率分布为边缘概率。

        离散型求解

        $$ P(x=x) = \sum _y P(x=x,y=y) $$

        连续型求解，用积分代替求和

        $$ p(x) = \int p(x,y)dy$$

    - 条件概率

        在给定其他事件时，求某个指定事件的发生概率。

        $$ P(y=y|x=x) = \dfrac {P(y=y,x=x)}{P(x=x)}$$

        P(x=x) > 0 才有意义，不能计算永不能发生的事件上的条件概率

        不同于计算一个行为的后果（干预查询/因果模型）。说德语的人有高概率可能是德国人，但并不是说找个会说德语的人，国籍会变为德国，国籍是不可变的。

    - 条件概率的链式法则

        任何多个随机变量的联合概率分布都可以分解为只有一个变量的条件概率相乘，又称为乘法法则

        $$ P(a,b,c) = P(a|b,c)P(b,c) $$
        $$ P(b,c) = P(b|c)P(c) $$
        $$ P(a,b,c) = P(a|b,c)P(b|c)P(c) $$
    
    - 独立性和条件独立性

        如果 x,y 的概率分布可以表示成两个因子的乘积，则两个变量是相互独立的。如果x,y的概率对于z的每一个值都可以分解为两个因子的乘积，则为条件独立的。

    - 期望、方差和协方差

        函数 f(x) 关于某分布 P(x) 的期望是当x由P产生，f作用于x时，f(x)的平均值。期望是线性的。

        方差是对x按照其概率分布进行采样时，f(x) 的值呈现多大的差异。方差越小，f(x) 越接近期望值。方差的平方根称为标准差。

        协方差，给出了两个变量线性相关的强度和变量的尺度。协方差的绝对值越大，意味变量值的变化很大，且距离各自的均值很远；如果协方差为正，两个变量都倾向于同时取得较大的值，如果为负，则一个变量为大值，一个变量为小值。如果两个变量相互独立，则协方差为0；

    - 常用概率分布
        - 伯努利分布
            离散分布，要么1，要么0，又称为2点分布。

            概率质量函数为：

            $$ fx(x)=p^x(1-p)^{(1-x)}=\begin{cases} p\qquad if\ x=1,\\1-p \qquad if\ x=0 \end{cases} $$
    
            期望值：

            $$ E[X]=\sum _{i=0}^1x_ifx(x)=0+p=p $$

            方差：

            $$ var[X]=\sum _{i=0}^1(x_i-E[X])^2=(0-p)^2(1-p)+(1-p)^2p = p(1-p) $$
    
        - Mutinoulli 分别
            在具有k个不同状态的单个离散型随机变量上的分布，k是有限值
            
            因为简单所以和伯努利分布常用

        - 高斯分布
            是连续概率分布，又称正态分布

            概率密度函数为：

            $$ f(x) = \dfrac {1}{\sigma\sqrt {2\pi}}e^{-\dfrac {(x-\mu)^2}{2\sigma^2}} $$

            $\mu$是位置参数，觉得分布的位置；方差$\sigma^2$的开平方或标准差$\sigma$等于尺度参数，决定了分布的幅度。

            标准正态分布，是 $\mu=0$ ,$\sigma^2=1$ 的正态分布。

            用的很多，主要大部分的真实情况都符合正态分布；另外具有相同方差的情况下，正态分布具有最大不确定性，所以对于先验知识的干扰最小。

        - 指数分布和拉普拉斯分布

            连续概率分布，x为负值时，概率为0

            概率密度函数为：

            $$ f(x;\lambda) = \begin{cases} \lambda e^{-\lambda x}\qquad if\ x\geq 0,\\0 \qquad if\ x<0 \end{cases}  $$

            $\lambda>0$是率参数，每单位时间发生该事件的次数。

            期望值：

            $$ E[X] = \dfrac {1}{\lambda} $$

            方差：

            $$ D[X] = \dfrac {1}{\lambda^2} $$

            拉普拉斯分布为双指数分布，尾端比正态分布更平缓

        - 狄拉克分布和经验分布

            狄拉克分布为1个点，除零以外的点上都是零，但积分是1。可以当做一个脉冲

            狄拉克分布通常作为经验分布的组成部分。经验分布函数是在所有n个数据点上都跳跃1/n的阶跃函数，阶梯状。

        - 分布的混合

            经验分布就是多个狄拉克分布的组成。混合模型引入了潜变量的概念。

            常见的混合模型是高斯混合模型，其是概率密度的万能近似器。任何平滑的概率密度都可以用足够多组件的高斯混合模型以任意精度来逼近。

        - 常用函数的有用性质

            - logistic sigmoid 函数
            
            $$ f(x) = \dfrac {1}{1+e^{-x}} $$

            最初是指数增长，后面变得饱和，对输入不敏感增长变慢，值区间0~1，当输入在[-1,1]之间时，函数变化敏感，一旦超过就饱和

            - tanh

            $$ f(x) = \dfrac {e^x-e^{-x}}{e^x+e^{-x}}$$

            近似于sigmoid函数，但后端能保持非线性变化，延迟了饱和期，用的多，sigmoid 已经少用，饱和区间很容易由于变化太小造成梯度消失。

            - relu 函数

            $$ f(x) = max(0,x) $$

            - softplus 函数

            $$ f(x) = log(1+e^x) $$

            softplus 是 relu 的平滑, 但relu会是一部分神经元输出为0，造成网络的稀疏性，减少过拟合。

            - softmax 函数

            $$ \sigma(z)_j= \dfrac {e^{z_j}}{\sum _{k=1}^Ke^{z_k}}$$

            也就是离散数据归一化，将任意向量压缩到(0,1)区间，并让所有元素之和为1. 特点是反向传播时求偏导简单快速。

        - 贝叶斯规则

            $$ P(x|y)=\dfrac {P(x)P(y|x)}{P(y)} $$

            上述公式是已知 P(y|x) 求 P(x|y)， $P(y)=\sum _xP(y|x)P(x)$ 计算。
            
            例子： 假设一个常规的检测结果的敏感度与可靠度均为99%，即吸毒者每次检测呈阳性（+）的概率为99%。而不吸毒者每次检测呈阴性（-）的概率为99%。从检测结果的概率来看，检测结果是比较准确的，但是贝叶斯定理却可以揭示一个潜在的问题。假设某公司对全体雇员进行吸毒检测，已知0.5%的雇员吸毒。请问每位检测结果呈阳性的雇员吸毒的概率有多高？

            令“D”为雇员吸毒事件，“N”为雇员不吸毒事件，“+”为检测呈阳性事件。可得

            - P(D)代表雇员吸毒的概率，不考虑其他情况，该值为0.005。因为公司的预先统计表明该公司的雇员中有0.5%的人吸食毒品，所以这个值就是D的先验概率。
            - P(N)代表雇员不吸毒的概率，显然，该值为0.995，也就是1-P(D)。
            - P(+|D)代表吸毒者阳性检出率，这是一个条件概率，由于阳性检测准确性是99%，因此该值为0.99。
            - P(+|N)代表不吸毒者阳性检出率，也就是出错检测的概率，该值为0.01，因为对于不吸毒者，其检测为阴性的概率为99%，因此，其被误检测成阳性的概率为1 - 0.99 = 0.01。       

            得： 

            $$ P(D|+) = \dfrac {P(+|D)P(D)}{P(+)} $$
            $$ = \dfrac {P(+|D)P(D)}{P(+|D)P(D)+P(+|N)P(N)} $$
            $$ = \dfrac {0.99 * 0.005} {0.99*0.005+0.01*0.995} $$
            $$ = 0.3322 $$

            尽管吸毒检测的准确率高达99%，但贝叶斯定理告诉我们：如果某人检测呈阳性，其吸毒的概率只有大约33%，不吸毒的可能性比较大。假阳性高，则检测的结果不可靠。

        - 连续型变量的技术细节

            零测度集在度量空间不占体积

            某性质几乎处处存在，是指在零测度集外都成立

            y=g(x),g是连续可微的函数，$P_y(y)=P_x(g^{-1}(y))$ 不成立，应该为$P_y(y)=P_x(g^{-1}(y))\left| \dfrac {\partial g\left( x\right) }{\partial x}\right|$

        - 信息论

            信息论的基本思路是，一个不太可能发生的事件比一个非常可能发生的事件能提供更多的信息。

            定义一个x=x自信息为：

            $$ I(x)= -log P(x) $$

            单位是奈特(nats)，1 奈特是以1/e的概率观测到一个事件时获得的信息量。有些地方用 2 为底数，称为比特或香农。

            一个分布的香农熵是指遵循这个分布的事件所产生的期望信息总量。

            如果x是连续的，香农熵又称为微分熵。

            相对熵又称为KL散度或KL距离，并不是真正的距离，是非对称的

            $$ D_{kl}(p||q) <> D_{kl}(q||p)   $$

            $$ D_{kl}(p||q) = \sum _{x\in X} p(x)log \dfrac {p(x)}{q(x)} = H_p(Q)-H(p)$$

            为了保证连续性，约定：

            $$ 0log\dfrac {0}{0}=0,\ 0log\dfrac {0}{q}=0,\ plog\dfrac{p}{0}=\infty$$

            交叉熵主要是用于度量两个概率分布间的差异性信息。相当于将相对熵中的H(p)视为0，也成为最小化KL距离。

            离散分布的交叉熵

            p:真实样本分布，q待估计的模型，参数服从(0-1))： 
            
            $$ CEH(p,q)=-\sum _{x\in X}p(x)log\ q(x) $$
            $$ = -[P_p(x=1)logP_q(x=1)+Pp(x=0)logP_q(x=0)] $$
            $$ = -[plogq+(1-p)log(1-q)] $$
            $$ = -[ylogh_\theta(x)+(1-y)log(1-h_\theta(x))] $$
            对所有样本取均值：

            $$ -\dfrac {1}{m}\sum _{i=1}^m[y^{(i)}logh_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$$


            连续分布的交叉熵：

            $$ -\int _x P(x)log\ Q(x)dr(x)=E_p[-logQ] $$

            限制：

            当 $p(x)\rightarrow 0$时，$p(x)log\ p(x)\rightarrow 0$

            例子：

            小明成绩差，考试经常不及格，小王成绩好，经常满分。

            事件A：小明考试及格的概率为$P(x_A)=0.1$，信息量为：$I(x_A)=-log(0.1)=3.3219$

            事件B：小王考试及格的概率为$P(x_B)=0.999$，信息量为：$I(x_B)=-log(0.999)=0.0014$

            假设小明的考试结果是0-1分布$x_A$只有2个值，0不及格，1及格，则某次考试的小明的及格概率只有0.1，计算所有的结果的熵的和为衡量小明考试结果的不确定度，即小明熵：

            $$H_A(X)=-(p(x_A)log(p(x_A))+(1-p(x_A))log(1-p(x_A)))=-(0.1*log(0.1)+0.9*log(0.9))=0.4690$$

            对应小王的熵为：

            $$H_B(X)=-(p(x_B)log(p(x_B))+(1-p(x_B))log(1-p(x_B)))=-(0.999*log(0.999)+0.001*log(0.001))=0.0114$$

            即小明的考试的预测结果的准确度比小王预测结果的准确度低。

            如果在有个学生小东的及格概率是0.5，则对应的熵为：

            $$H_C(X)=-(p(x_C)log(p(x_C))+(1-p(x_C))log(1-p(x_C)))=-(0.5*log(0.5)+0.5*log(0.5))=1$$

            即，预测小东成绩的准确性更差，基本没法预测。

            机器学习中的交叉熵

            线性回归，用MSE,均方误差:

            $$ loss = \dfrac {1}{m}\sum _{i=1}^m(p_i-q_i)^2 $$

            逻辑回归，用交叉熵，因为小概率更有信息价值，符合人脑学习方式：

            单分类问题：

            $$ loss = -\dfrac {1}{m}\sum _{i=1}^mp_ilog(q_i) $$

            | * | 猫 | 青蛙 | 狗 |
            | ------ | ------ | ------ | ------ |
            | Label(p) | 0 | 1 | 0 |
            |  Pred(q) | 0.3 | 0.6 | 0.1|

            那么：

            $$loss = -(0*log(0.3)+1*log(0.6)+0*log(0.1))=-log(0.6)=0.7370$$
            
            多批量的loss为所有loss的平均值。

            多分类问题，这里的Pred不是采用softmax计算，采用sigmoid，将每一个节点都归一化到[0,1]之间，由于每个节点只有两种可能，所以是一个二项分布，则简化交叉熵计算：

            $$ loss = -plog(q)-(1-p)log(1-q) $$

            | * | 猫(A) | 青蛙(B) | 狗(C) |
            | ------ | ------ | ------ | ------ |
            | Label(p) | 0 | 1 | 1 |
            |  Pred(q) | 0.1 | 0.7 | 0.8|

            那么：

            $$loss_A = -0*log(0.1)-(1-0)log(1-0.1)=-log(0.9)=0.1520$$
            $$loss_B = -1*log(0.7)-(1-1)log(1-0.7)=-log(0.7)=0.5146$$
            $$loss_C = -1*log(0.8)-(1-1)log(1-0.8)=-log(0.8)=0.3219$$
            $$loss=loss_A+loss_B+loss_C=0.9985$$

            多批量的loss为所有loss的平均值。

        - 结构化概率模型

            如果有3个变量，a、b、c，a影响b，b影响c，但b给定时，a、c之间独立。因此可以将联合概率改为2个变量概率的乘积。

            $$ p(a,b,c) = p(a)p(b|a)p(c|b) $$

            用图来表示这个概率分布的分解时，就称为结构化概率模型或图模型

            分为有向和无向两种

4. 数值计算

    1. 上溢和下溢

        $softmax(x)_i=\dfrac {e^{x_i}}{\sum _{j=1}^ne^{x_j}}$ 如果 $x_j$的值很小的负数，则$e^{x_j}$为0，则分母为0，导致值无意义；如果$x_i$为大的正数时，$e^{x_i}$ 趋于无穷大，超界。

        这两个问题可以采用 softmax(z) 解决，$z=x-max_ix_i$ 

        例如计算 [3,1,-3] 中间 1 的softmax:

        传统方法：

        $$ softmax(1) = \dfrac {e^1}{e^3+e^1+e^{-3}} = \dfrac {2.7}{20+2.7+0.05} = 0.12 $$

        新方法：
        
        $$ M = max([3,1,-3]) = 3 $$

        $$ softmax(1) = \dfrac {e^{1-M}}{e^{3-M}+e^{1-M}+e^{-3-M}} $$
        $$ = \dfrac {e^{1-3}}{e^{3-3}+e^{1-3}+e^{-3-3}} = \dfrac {0.1353}{1+0.1353+0.0025} = 0.12 $$

        以上分子最大0，分母最小1，所以同时解决了上溢和下溢的问题。

        下一步求 log(softmax(x)) ，softmax(x) 有可能为0，导致 log(0) 无意义。 

        解决办法：

        $$ log[f(x_i)]=log(\dfrac{e^{x_i}}{e^{x_1}+e^{x_2}+...e^{x_n}})  $$
        $$ =log(\dfrac{ \dfrac{e^{x_i}}{e^M} }{ \dfrac{e^{x_1}}{e^M}+\dfrac{e^{x_2}}{e^M}+...\dfrac{e^{x_n}}{e^M} })  $$
        $$ =log(\dfrac {e^{(x_i-M)}} {\sum _j^ne^{(x_j-M)} }) $$
        $$ =log({e^{(x_i-M)}}) - log(\sum _j^ne^{(x_j-M)}) $$
        $$ = (x_i-M)-log(\sum _j^ne^{(x_j-M)}) $$

        可以看到log的求和项最小为1，就解决了log项的下溢问题。

    1. 病态条件

        条件数是函数相对于输入的微小变化而变化的快慢程度。输入轻微而函数变化剧烈会有问题。

        病态条件的矩阵会放大预先存在的错误，这个错误将会和反向算法的数值误差进一步复合。

    1. 基于梯度的优化方法

        优化方法是改变x以最小化或最大化f(x)的任务，一般都是最小化f(x),最大化可以采用-f(x)来实现。

        需要最小化函数称为目标函数或准则。但最小化时，也称为代价函数、损失函数或误差函数。

        一般用上标 * 表示最小化函数，例如：$x^*=argmin\ f(x)$

        - 微积分与优化

            - 定义 $y=f(x)$,其中x、y都是实数，这个函数的导数为 $f'(x)$ 或 $\dfrac {d_y}{d_x}$ , 导数 f‘(x) 代表 f(x) 在 x 点上的斜率。

            - 导数对最小化的用处在于，告诉我们如何更改x来略微的改善y。我们将x往导数的反方向移动一小步来减小f(x),这种技术成为梯度下降。

            - 当导数 f'(x)=0 时，导数无法提供移动信息，这个点称为临界点或驻点。一个局部极大点说明不管如何移动，f(x)都比所有临近点大。有些临界点不是最小点也不是最大点，称为鞍点。

            - f(x) 绝对的最小值的点，称为全局最小点。当出现多个局部极小点或平坦区域时，优化函数将有可能无法找到全剧最小点。

            - 如果有多维输入的情况，则采用偏导数。偏导数$\dfrac {\partial }{\partial x_{i}}f(x)$ 衡量点x处只有$x_i$增加时f(x)如何变化。梯度是相对一个向量求导的导数，f的导数是包含所有偏导数的向量，记做：$\nabla _xf(x)$, 梯度的第i个元素是f关于$x_i$的偏导数。在多维情况下，临界点是梯度中所有元素都为0的点。

            - 在u(单位向量)方向的方向导数，是函数f在u方向的斜率。方向导数是函数 $f(x+\alpha u)$关于$\alpha$的导数（在$\alpha=0$时取得）。根据链式法则，当$\alpha=0$时， $\dfrac {\partial }{\partial x_{i}}f(x+\alpha u)=u^T\nabla _xf(x)$, 为了最小化f，找到使f下降最快的方向，计算方向导数：

            $$ \min _{u,u^Tu=1} u^T\nabla _xf(x) $$
            $$ = \min _{u,u^Tu=1} ||u||_2||\nabla _xf(x)||_2cos\theta $$

            $\theta$是u与梯度的夹角，将$||u||_2=1$代入，可以简化为 $\min _{u} cos\theta$

            - 我们可以在负梯度上移动可以减小f，这个称为最速下降法或梯度下降。

            - 最速下降的新的点为 $x'=x-\varepsilon \nabla _xf(x)$,其中$\varepsilon$ 为学习率，是一个确定步长大小的正标量。这个值通常是一个小参数，可以通过实际测试能产生最小化目标函数来找到，这个策略称为线搜索。

            - 最速下降到每一个元素都为0或接近为0时，可以尝试直接解方程 $\nabla _xf(x)=0$得到临界点。

        - 梯度之上：雅可比和海森矩阵

            所有输入和输出都为向量的函数的偏导数矩阵称为雅可比矩阵(Jacobian)。

            定义：$f:\ \mathbb{R}^m \rightarrow \mathbb{R}^n$ 

            则雅可比矩阵J为： $J \in \mathbb{R}^{nxm}$ ,$J_{i,j}=\dfrac {\partial}{\partial x_j}f(x)_i$

            导数的导数称为二阶导数

            一阶导数是衡量输入导致的变化率，二阶导数是衡量曲率。如果曲率为0，则代价函数符合预期的下降速度；如果是负曲率，则代价函数实际上比预测的下降的更快；正曲率，代价函数比预计的下降更慢，并且最终会增加，这时如果太大的步骤会导致增加函数值。

            如果是多维的情况，所有的维度的二阶导数组成的矩阵，称为海森矩阵(Hessian)。

            定义： $H(f)(x)_{i,j}=\dfrac {\partial^2}{\partial x_i\partial x_j}f(x)$

            海森矩阵等于梯度的雅可比矩阵。

            可以通过二阶导数预期梯度下降步骤的表现和计算最优步长。还可以用二阶导数确定一个临界点是否是局部最大点、局部最小点或鞍点。当二阶导数特征值全部为正，则为局部最小点；全部为负，则为局部最大点；除外为鞍点。

            如果梯度下降表现很差时，可以利用牛顿法，利用海森矩阵来指导下降。

            牛顿法先假设任务是优化一个目标函数f，求函数的极小问题，可以转换为求解函数f的导数 f'=0 的问题，如下：

            为了求解 f'=0 的根，利用泰勒公式把f(x)在$X_n$展开到二阶，即：

            $$f(x)\approx f(x_n)+f'(x_0)(x-x_n)+\dfrac {f''(x_n)}{2}(x-x_n)^2$$

            然后用f(x)的最小点作为新的探索点$x_{n+1}$，据此，令：

            $$ f'(x) = f'(x_n) + f''(x_n)(x-x_n) = 0 $$

            求出迭代公式，即：

            $$x_{n+1}=x_n-\dfrac {f'(x_n)}{f''(x_n)},\ n=0,1,...$$

            高维度的牛顿迭代公式为：

            $$x_{n+1}=x_n-Hf(x_n)^{-1}\nabla f(x_n),\ n \geq 0 $$

            一般认为牛顿法比梯度下降法利用了曲率信息，更容易收敛。

            仅仅利用梯度信息的优化算法称为一阶优化算法，例如梯度下降；使用海森矩阵的优化算法称为二阶最优化算法，如牛顿法。

            在限定领域最成功的优化为凸优化，但只对凸函数有效。但深度学习中的很难表示成凸优化的形式，凸优化只能作为深度学习算法中的子算法。目前深度学习中的各种优化算法，由于使用的函数簇非常复杂，所以大多数缺乏理论保证。但限制函数利普希茨(Lipschitz)连续或其导数利普希茨连续，可以获得一些保证。利普希茨连续要求函数曲线上的任意两点的斜率一致有界，就是任意斜率都小于同一个常数，这个常数称为利普希茨常数。这个属性可以量化训练假设，即梯度下降算法产生的输入的微小变化导致输出也是微小变化。反例 $f(x) = \sqrt {x}$这个函数两点间的斜率可以无限大，因此不是利普希茨连续。

    1. 约束优化

        希望在x的某些集合$\mathbb {S}$中找到f(x)的最大值或最小值，这个称为约束优化。集合$\mathbb {S}$中的点x称为可行点。例如对 x 加一个范数约束 $||x|| \leq 1$。

        简单的做法修改梯度下降的步长或将线上的每一个点投影到约束区域。

        或设计一个不同的、无约束的优化问题，其解可以转化为原始约束的解。如：求解有x约束为L2范数的解，可以转换为求解最小化 $\theta$, $\theta = f(|cos\theta,sin\theta|^T)$ ，最后返回 $[cos\theta,sin\theta]$ 做为原始 L2 范数约束的解。 

        还可以使用卡罗需-库恩-塔克条件（KKT）来求解。

        假设目标函数$f:\mathbb {R}^n\rightarrow \mathbb {R}$及约束函数$g_i:\mathbb {R}^n\rightarrow \mathbb {R}$皆为凸函数，而$h_i:\mathbb {R}^n\rightarrow \mathbb {R}$是一仿射函数，假设有一可行点$x^*$，如果有常数$\mu _i \geq 0(i=1,...m)$及$v_j(j=1,...,l)$令到：

        $$\nabla _xf(x^*)+\sum _{i=1}^m\mu _i\nabla g_i(x^*)+\sum _{j=1}^lv_j \nabla h_j(x^*)=0$$
        $$\mu _ig_i(x^*)=0\ for\ all\ i=1,...,m,$$
        那么$x^*$这点就是全局极小值。

    1. 实例：线性最小乘

        假设，我们找到下面x的最小化值：

        $$f(x)=\dfrac {1}{2}||Ax-b||_2^2$$
        - 定义：

        $$||x||_2^2=\sqrt {(\sum _{i=1}^mx_i^2)}=\sqrt {x^Tx}$$

        - 求解：

        $$f(x)=\dfrac {1}{2}(Ax-b)^T(Ax-b)$$
        $$=\dfrac {1}{2}(x^TA^T-b^T)(Ax-b)$$
        $$=\dfrac {1}{2}(x^TA^TAx-2b^TAx+b^Tb)$$
        注：b、x都是列向量，所以$b^TAx$是标量，标量的转置等于自身，即：$b^TAx=X^TA^Tb$

        - 对x求导，得到梯度：

        $$\nabla _x f(x)=A^TAx-A^Tb=A^T(Ax-b)$$

        - 最小二乘法计算：
        >将步长($\varepsilon$)和容差($\delta$)设置为小的正数
        
        > $while\ ||A^TAx-A^Tb||_2>\delta\ do$
        >
        > $\qquad x\leftarrow - \varepsilon(A^TAx-A^Tb)$
        >
        > $end\ while$

    1. 详细实例，最小二乘法：

        用3个温度计测量当前温度，求平均温度：
        |  | 温度计1 | 温度计2 | 温度计3 |
        | ------ | ------ | ------ | ------ |
        | 温度 | 29.7 | 28.7 | 30.1 |

        常规解法，求平均值 $\overline C = \dfrac {(C1+C2+C3)}{3} = 29.5$

        最小二乘解法：

        $$S=\sum _{i=1}^3(C-C_i)^2$$

        S为种误差值，我们需要最小化S，求C。两边求导：

        $$\dfrac {dS}{dC}=0=\dfrac {d\sum _{i=1}^3(C-C_i)^2}{dC}$$
        $$=\dfrac {d\sum _{i=1}^3(C^2-2CC_i+C_i^2)}{dC}$$
        $$= \sum_{i=1}^32C-2C_i $$
        $$= 2\sum_{i=1}^3(C-C_i)$$
        $$= 2((C-C_1)+(C-C_2)+(C-C_3))$$
        $$= 2(3C-(C_1+C_2+C_3))$$
        
        求解得：

        $$ 2(3C-(C_1+C_2+C_3))=0 $$
        $$ C = \dfrac {C_1+C_2+C_3}{3}=29.5$$

        已知温度和冰淇淋的销量，预测某一温度下销量：
        | 温度 | 31 | 33 | 35 |
        | ------ | ------ | ------ | ------ |
        | 销量 | 113 | 119 | 123 |

        看上去是线性关系，先假定一个线性方程来匹配：

        $$f(x)=ax+b$$

        定义误差函数为：

        $S=\sum _{i=1}^3(ax_i+b-y_i)^2$

        $y_i$是同一温度下预测的值。x是温度，ax+b是预测销售值。目标是最小化S

        由于有2个未知数a、b，分别求偏导：

        $$\begin{aligned}
        \dfrac {\partial S}{\partial a}&=\dfrac {d\sum((ax_i)^2+2ax_i(b-y_i)+(b-y_i)^2)}{da}  \\\\
        &=\sum 2ax_i^2+2x_i(b-y_i) \\\\
        &=2\sum(ax_i+b-y_i)x_i=0  
        \end{aligned}$$

        $$\begin{aligned}
        \dfrac {\partial S}{\partial b}&=\dfrac {d\sum(ax_i-y_i+b)^2}{db} \\\\
        &=\dfrac {d\sum((ax_i-y_i)^2+2(ax_i-y_i)b+b^2)}{db} \\\\
        &=\sum (2(axi-y_i)+2b) \\\\
        &=2\sum(ax_i+b-y_i)=0  
        \end{aligned}$$

        列方程为：
        $$
        \begin{cases}
        2((31a+b+113)*31+(33a+b+119)*33+(35a+b+123)*35)=0\\\\
        2((31a+b-113)+(33a+b-119)+(35a+b-123))=0
        \end{cases}
        $$

        解方程得：

        $$y=2.5x+35.83$$

        我们还可以假设方程为如下方程：

        $$f(x)=ax^2+bx+c$$

        或

        $$f(x)=a/x+b$$

        或

        $$f(x)=ae^{(x+b)}$$
        
        也一样可以做线性二乘法。

        最小二乘法求解的值满足误差的正态分布。如果某些场合不满足正态分布，也可以通过对数据取对数得到正态分布的结果。












        












