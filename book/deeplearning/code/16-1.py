import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import os

# 受限玻尔兹曼机（RBM）
# 可以用于降维、特征提取和协同过滤，RBM 的训练可以分成三部分：正向传播、反向传播和比较。
class RBM(object):
    def __init__(self, m, n):
        '''
        m: 可见层的神经元个数
        n: 隐藏层的神经元个数
        '''
        self._m=m
        self._n=n
        # 权重和偏置 _c 隐藏层偏置， _b 可见层偏置
        self._W=tf.Variable(tf.random_normal(shape=(self._m,self._n)))
        self._c=tf.Variable(np.zeros(self._n).astype(np.float32))
        self._b=tf.Variable(np.zeros(self._m).astype(np.float32))
        # 输入
        self._X=tf.placeholder('float',[None,self._m])
        # 前向传播
        _h = tf.nn.sigmoid(tf.matmul(self._X,self._W)+self._c)
        self.h=tf.nn.relu(tf.sign(_h-tf.random_uniform(tf.shape(_h))))
        # 反向传播
        _v = tf.nn.sigmoid(tf.matmul(self.h,tf.transpose(self._W))+self._b)
        self.v=tf.nn.relu(tf.sign(_v-tf.random_uniform(tf.shape(_v))))
        # 目标函数
        objective = tf.reduce_mean(self.free_energy(self._X))-tf.reduce_mean(self.free_energy(self.v))
        self._train_op= tf.train.GradientDescentOptimizer(1e-3).minimize(objective)
        # 损失函数
        reconstructed_input = self.one_pass(self._X)
        self.cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self._X,logits=reconstructed_input))

    # 批训练
    def fit(self, X, epochs =1, batch_size=100):
        N,D=X.shape
        num_batches = N//batch_size
        obj=[]
        for i in range(epochs):
            for j in range(num_batches):
                batch = X[j*batch_size:(j*batch_size+batch_size)]
                _,ob=self.session.run([self._train_op,self.cost],feed_dict={self._X:batch})
                if j%10==0:
                    print('training epoch {0} cost {1}'.format(j, ob))
                obj.append(ob)
        return obj

    def set_session(self, session):
        self.session = session
    
    def free_energy(self, V):
        b = tf.reshape(self._b, (self._m,1))
        term_1 = -tf.matmul(V,b)
        term_1 = tf.reshape(term_1, (-1,))
        term_2 = -tf.reduce_sum(tf.nn.softplus(tf.matmul(V,self._W)+self._c))
        return term_1 + term_2

    def one_pass(self, X):
        h = tf.nn.sigmoid(tf.matmul(X,self._W)+self._c)
        return tf.matmul(h, tf.transpose(self._W))+self._b

    def reconstruct(self, X):
        x = tf.nn.sigmoid(self.one_pass(X))
        return self.session.run(x, feed_dict={self._X:X})
    
if __name__ == '__main__':
    curr_dir = os.path.dirname(__file__)
    mnist = input_data.read_data_sets(os.path.join(curr_dir,"../data"), one_hot=True)
    trX,trY,teX,teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

    Xtrain = trX.astype(np.float32)
    Xtest = teX.astype(np.float32)
    _,m = Xtrain.shape
    rbm = RBM(m, 500)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        rbm.set_session(sess)
        err = rbm.fit(Xtrain)
        out = rbm.reconstruct(Xtest[0:100])

    row, col = 2,8
    idx = np.random.randint(0, 100, row*col//2)
    f, axarr = plt.subplots(row, col, sharex=True, sharey=True, figsize=(20,4))
    for  fig, row in zip([Xtest, out], axarr):
        for i,ax in zip(idx,row):
            ax.imshow(fig[i].reshape((28,28)),cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()