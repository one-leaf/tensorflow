# 多项式和正则化
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 样本个数
count=8
# 多项式的项数
n=12
# 是否开启w正则
add_L2_Loss = False

# 从0~1之间取10个点
x_train = np.linspace(0, 1, count, dtype=np.float32)
# 定义 y 的函数输出，同时加上一个正态分布的干扰点
# 这个函数对于最小二乘法而言是未知的
y_train = np.sin(2 * np.pi * x_train) + np.random.normal(0, 0.1, (count,))

# 定义模型输入
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 定义参数
w=[tf.Variable(tf.random_uniform([1], -1, 1)) for i in range(n)]
b = tf.Variable(tf.zeros([1]))

# 定义预测函数为多项式和
y_pred = 0
for i in range(n):
    y_pred = y_pred + tf.multiply(w[i], tf.pow(x, i))
y_pred += b

# 定义损失函数
loss = tf.reduce_mean(tf.square(y-y_pred))

# 对w参数引入正则项
if add_L2_Loss:
    loss +=  tf.add_n([ tf.nn.l2_loss(v) for v in w ]) * 0.0001

# 定义梯度下降法优化函数，优化
optimizer = tf.train.GradientDescentOptimizer(0.3)
# 定义训练目标是采用梯度下降法最小化损失函数
train = optimizer.minimize(loss)

# 初始化 tensorflow 运行环境
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 开始训练
for step in range(20000):
    _, tmp_loss = sess.run([train, loss], feed_dict={x:x_train, y:y_train})
    if step % 1000 == 0:
        print(step, tmp_loss) 

# 展示数据
x_test = np.linspace(0, 1, 1000, dtype=np.float32)
y_test = np.sin(2 * np.pi * x_test)
y_test_pred = sess.run(y_pred, feed_dict={x:x_test})
plt.plot(x_test, y_test, 'b', label='real data')
plt.plot(x_train, y_train, 'bo', label='train data')
plt.plot(x_test, y_test_pred, 'r', label='predicted data')
plt.legend()
plt.show()