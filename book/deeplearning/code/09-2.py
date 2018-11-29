# 简单 CNN
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

    # 增加卷积层
    def add_conv_layer(self, inputs, filter_size, out_size, activation_function=None, pool_function=None):
        shape = inputs.get_shape().as_list()
        Weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, shape[-1], out_size], stddev=0.1))
        biases = tf.Variable(tf.zeros([out_size]) + 0.1)
        layer = tf.nn.conv2d(inputs, Weights, strides=[1, 1, 1, 1], padding='VALID')
        output = layer + biases
        if activation_function != None: 
            output = activation_function(output)
        if pool_function != None:
            output = pool_function(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return output

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        # 3层cnn单元
        x_image = tf.reshape(self.x, [-1,28,28,1])
        layer1 = self.add_conv_layer(x_image, 5, 16, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 
        layer2 = self.add_conv_layer(layer1, 5, 32, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 
        layer3 = self.add_conv_layer(layer2, 3, 64, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 

        # 扁平化，转全连接层做分类输出
        shape = layer3.get_shape().as_list()
        layer_size = shape[1]*shape[2]*shape[3]
        layer = tf.reshape(layer3, [-1,layer_size])

        w = tf.Variable(tf.random_normal([784, layer_size]))
        b = tf.Variable(tf.zeros([layer_size])+0.1)
        hide_layer = tf.nn.relu(tf.add(tf.matmul(self.x, w), b))

        layer = tf.concat([layer,hide_layer],-1)
        print(layer.shape)

        w = tf.Variable(tf.random_normal([layer_size*2, 10]))
        b = tf.Variable(tf.zeros([10])+0.1)
        self.full_connect_layer = tf.add(tf.matmul(layer, w), b)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.full_connect_layer))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.full_connect_layer,1),tf.argmax(self.y,1)),tf.float32))
        self.optimizer= tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)
    
def main():
    loss_list=[]
    net = network()
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
                loss_list.append(loss_totle)
            acc = net.accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print(epoch, "cross_entropy:", loss_list[-1],"acc:", acc)

    plt.figure()
    x = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x, loss_list, label='acc: %s loss'%acc, color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

