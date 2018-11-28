# 简单 CNN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# 导入手写体数据
from tensorflow.examples.tutorials.mnist import input_data
curr_path = os.path.dirname(os.path.realpath(__file__))
mnist = input_data.read_data_sets(os.path.join(curr_path,"../data"), one_hot=True)

# 定义神经网络
class network():

    # 增加卷积层
    def add_conv_layer(self, inputs, filter_size, out_size, activation_function=None, pool_function=None):
        shape = inputs.get_shape().as_list()
        Weights = tf.Variable(tf.truncated_normal([filter_size, filter_size, shape[-1], out_size], stddev=0.1))
        biases = tf.Variable(tf.zeros([out_size]) + 0.1)
        layer = tf.nn.conv2d(inputs, Weights, strides=[1, 1, 1, 1], padding='VALID')
        Wconvlayer_plus_b = layer + biases
        if activation_function is None:
            convlayer = Wconvlayer_plus_b
        else:
            convlayer = activation_function(Wconvlayer_plus_b)
        if pool_function is None:
            outputs = convlayer
        else:
            outputs = pool_function(convlayer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return outputs

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        # 3层cnn单元
        x_image = tf.reshape(self.x, [-1,28,28,1])
        layer1 = self.add_conv_layer(x_image, 5, 4, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 
        layer2 = self.add_conv_layer(layer1, 3, 8, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 
        layer3 = self.add_conv_layer(layer2, 3, 16, activation_function=tf.nn.relu, pool_function=tf.nn.max_pool) 

        # 扁平化，转全连接层做分类输出
        shape = layer3.get_shape().as_list()
        layer_size = shape[1]*shape[2]*shape[3]
        layer = tf.reshape(layer3, [-1,layer_size])
        w = tf.Variable(tf.random_normal([layer_size, 10]), name="weights")
        b = tf.Variable(tf.zeros([10]), name="bias")
        self.full_connect_layer = tf.add(tf.matmul(layer, w), b)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.full_connect_layer))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.full_connect_layer,1),tf.argmax(self.y,1)),tf.float32))
        self.optimizer= tf.train.AdamOptimizer(0.01).minimize(self.cross_entropy)
    
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
            print(epoch, "cross_entropy:", loss_list[-1])
        acc = net.accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
        print("accuracy", acc)       

    plt.figure()
    x = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x, loss_list, label='acc: %s loss'%acc, color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

