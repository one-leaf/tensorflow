# 测试批标准化
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

class dateset():
    def __init__(self,images,labels):
        self.num_examples=len(images)                   # 样本数量
        self.images=np.reshape(images/255.,[-1,28*28])  # 图片归一化加扁平化
        self.labels=np.eye(10)[labels]                  # 标签 one-hot 化
    def next_batch(self, batch_size):                   # 随机抓一批图片和标签
        batch_index = np.random.choice(self.num_examples, batch_size)
        return self.images[batch_index], self.labels[batch_index]
class mnist():
    def __init__(self):
        # 导入mnist手写数据，x shape: (?,28,28); y shape: (?); x value: 0~255; y value: 0~9
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.train=dateset(x_train, y_train)
        self.test=dateset(x_test, y_test)

# 导入手写数据集
mnist = mnist()

# 定义神经网络
class network():

    # 增加 batch_normalization
    def batch_norm(self, input, training, eps=1e-05):
        ema = tf.train.ExponentialMovingAverage(decay=0.5)  
        input_shape = input.get_shape().as_list() 
        axes = list(range(len(input_shape) - 1))

        pop_mean, pop_var = tf.nn.moments(input, axes, name='moments')
        def mean_var_with_update():
            ema_apply_op = ema.apply([pop_mean, pop_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(pop_mean), tf.identity(pop_var)
        mean, variance = tf.cond(training, mean_var_with_update, lambda: (ema.average(pop_mean), ema.average(pop_var)))
        beta = tf.Variable(initial_value=tf.zeros(input_shape[-1]))
        gamma = tf.Variable(initial_value=tf.ones(input_shape[-1]))
        output  = tf.nn.batch_normalization(input, mean, variance, beta, gamma, eps)
        return output 

    # 增加层
    def add_layer(self, input, out_size, batch_normalization=False):
        _, in_size = input.get_shape().as_list() 
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([out_size]) + 0.1)
        Wx_plus_b = tf.matmul(input, Weights) + biases

        if batch_normalization: # 批标准化
            Wx_plus_b = self.batch_norm(Wx_plus_b, self.training)

        return tf.nn.relu(Wx_plus_b)

    def __init__(self, layer_count=10, batch_normalization=False):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.training = tf.placeholder_with_default(True, shape=(), name='training')

        layer = self.x
        for i in range(layer_count):
            layer = self.add_layer(layer,64,batch_normalization)

        w = tf.Variable(tf.random_normal([64, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.full_connect_layer = tf.add(tf.matmul(layer, w), b)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.full_connect_layer))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.full_connect_layer,1),tf.argmax(self.y,1)),tf.float32))
        self.optimizer= tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)

def main():
    loss_dict={}
    acc_dict={}
    for normal in [True, False]:
        loss_dict[normal]=[]
        net = network(5, normal)
        print("batch_normalization:",normal)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 100
            loss_totle = 0 
            for epoch in range(10):
                total_batch = int(mnist.train.num_examples / batch_size)
                for step in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    loss,_= sess.run([net.cross_entropy,net.optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})
                    if loss_totle==0:
                        loss_totle=loss
                    else:
                        loss_totle=loss_totle*0.99+0.01*loss
                    loss_dict[normal].append(loss_totle)
                print(epoch, "cross_entropy:", loss_dict[normal][-1])

            acc_dict[normal]= net.accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels, net.training: False})
            print("accuracy", acc_dict[normal])       

    plt.figure()
    c = iter(cm.rainbow(np.linspace(0, 1, 2)))
    for normal in [True, False]:
        color = next(c)
        y = loss_dict[normal]
        x = np.linspace(0, len(y), len(y))
        plt.plot(x, y, label='BN: %s (acc: %s) loss'%(normal, acc_dict[normal]), color=color)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

