# 常用 CNN 模型，需要 GPU 支持
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim

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
    def __init__(self, network):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.training = tf.placeholder_with_default(True, shape=(), name='training')

        inputs = tf.reshape(self.x, [-1,28,28,1]) 
        inputs = tf.image.resize_images(inputs, (224, 224))     
        if network=="resnet50":
            with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
                net, endpoints = nets.resnet_v2.resnet_v2_50(inputs, num_classes=1000, is_training=self.training)
        elif network=="vgg19":
            with slim.arg_scope(nets.vgg.vgg_arg_scope()):
                net, endpoints = nets.vgg.vgg_19(inputs, num_classes=1000, is_training=self.training)
        elif network=="inception":
            with slim.arg_scope(nets.inception.inception_v3_arg_scope()):
                net, endpoints = nets.inception.inception_v3(inputs, num_classes=1000, is_training=self.training)
        elif network=="alexnet":
            with slim.arg_scope(nets.alexnet.alexnet_v2_arg_scope()):
                net, endpoints = nets.alexnet.alexnet_v2(inputs, num_classes=1000, is_training=self.training)
        else:
            raise Exception("UNKOWN MODLE %s"%network)
        
        net=slim.flatten(net)
        self.full_connect_layer= slim.fully_connected(net, num_outputs=10, activation_fn=None)
        
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.full_connect_layer))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.full_connect_layer,1),tf.argmax(self.y,1)),tf.float32))       
        self.optimizer = slim.learning.create_train_op(self.cross_entropy, tf.train.AdamOptimizer(0.001))

def main():
    loss_list=[]
    net = network("resnet50")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 50
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
                if step % 10 == 0:
                    test_xs, test_ys =  mnist.test.next_batch(batch_size)
                    acc = net.accuracy.eval({net.x: test_xs, net.y: test_ys, net.training: False})
                    print(epoch, "cross_entropy:", loss_list[-1],"acc:", acc)

    plt.figure()
    x = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x, loss_list, label='acc: %s loss'%acc, color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

