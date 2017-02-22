# tensorflow
一些机器学习的实践 [官网地址](https://www.tensorflow.org)

## 安装 ##
[windows下安装指南](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#pip-installation-on-windows)

1. 安装 python 3.5 64位，不能安装 python 3.6 目前不支持。

2. 安装 Visual C++ 2015 的运行库

3. 通过 pip install tensorflow 命令安装。

## 常用函数 ##

1. tf.nn.conv2d() 通过输入规定的四维input和filter计算二维卷积.

2. tf.nn.max_pool() 对input进行池化

3. tf.placeholder 插入一个待初始化的张量占位符

4. tf.constant 创建一个张量常量

5. tf.truncated_normal 从一个正态分布片段中输出随机数值

6. tf.random_normal 生成正太分布随机数

7. tf.random_uniform 均匀分布随机数

8. tf.matmul 将矩阵相乘

9. tf.reduce_mean 跨越维度的计算张量各元素的平均值

10. tf.reshape 就是将tensor按照新的shape重新排列

11. tf.square 对所有元素进行平方操作

12. tf.nn.dropout 按概率来将x中的一些元素值置零，并将其他的值放大

13. tf.ones | tf.zeros | tf.fill 生成矩阵，包括全1矩阵，全0矩阵和指定值

14. tf.expand_dims 为张量+1维

15. tf.pack 将一个R维张量列表沿着axis轴组合成一个R+1维的张量

16. tf.concat 将张量沿着指定维数拼接起来

17. tf.sparse_to_dense 稀疏矩阵转密集矩阵 

18. tf.random_shuffle 沿着value的第一维进行随机重新排列

19. tf.argmax | tf.argmin 找到给定的张量tensor中在指定轴axis上的最大值/最小值的位置。

20. tf.equal 判断两个tensor是否每个元素都相等。返回一个格式为bool的tensor

21. tf.cast 将x的数据格式转化成dtype

22. tf.nn.embedding_lookup 将一个数字序列ids转化成embedding序列表示

23. tf.trainable_variables 返回所有可训练的变量 

24. tf.gradients 计算导数

25. tf.clip_by_global_norm 修正梯度值，用于控制梯度爆炸的问题

26. tf.linspace | tf.range 产生等差数列

27. tf.assign 给变量赋值


## 常用链接 ##

[浏览器中运行ANN](http://datahref.com/sub/demo/ann/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.78579&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)