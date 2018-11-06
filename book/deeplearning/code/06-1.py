import numpy as np
import tensorflow as tf

x = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32) 
y = np.array([[0],[1],[1],[0]], dtype=np.float32)

W = tf.Variable(tf.random_uniform([2,2], -1, 1)) 
c = tf.Variable(tf.zeros([2], dtype=tf.float32))
w = tf.Variable(tf.random_uniform([2,1], -1, 1)) 
b = tf.Variable(tf.zeros([2], dtype=tf.float32))

Wx_plus_c=tf.matmul(x, W)+c
y_pred=tf.matmul(tf.nn.relu(Wx_plus_c), w)+b

# 定义损失函数
loss = tf.reduce_mean(tf.square(y-y_pred))


# 定义梯度下降法优化函数，优化
optimizer = tf.train.GradientDescentOptimizer(0.01)
# 定义训练目标是采用梯度下降法最小化损失函数
train = optimizer.minimize(loss)

# 初始化 tensorflow 运行环境
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 开始训练
for step in range(10000):
    _, tmp_loss = sess.run([train, loss])
    if step % 1000 == 0:
        print(step, tmp_loss) 

print('W:',sess.run(W))
print('c:',sess.run(c))
print('w:',sess.run(w))
print('b:',sess.run(b))
