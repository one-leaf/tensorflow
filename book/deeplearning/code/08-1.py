# 反向传播1
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
    def __init__(self, loss_optimizer):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.h_w = tf.Variable(tf.random_uniform([784, 784],-1,1), name="hide_weights")
        self.h_b = tf.Variable(tf.zeros([784]), name="hide_bias")
        self.hide_layer = tf.nn.relu(tf.add(tf.matmul(self.x, self.h_w), self.h_b))
        self.w = tf.Variable(tf.random_uniform([784, 10],-1,1), name="weights")
        self.b = tf.Variable(tf.zeros([10]), name="bias")
        self.full_connect_layer = tf.add(tf.matmul(self.hide_layer, self.w), self.b)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.full_connect_layer))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.full_connect_layer,1),tf.argmax(self.y,1)),tf.float32))
        self.optimizer= loss_optimizer.minimize(self.cross_entropy)
    
def main():
    learning_rate = 0.001
    optimizeres = {
        'GradientDescentOptimizer':tf.train.GradientDescentOptimizer(learning_rate),
        'AdagradOptimizer':tf.train.AdagradOptimizer(learning_rate),
        'MomentumOptimizer':tf.train.MomentumOptimizer(learning_rate, momentum=0.9),
        'RMSPropOptimizer':tf.train.RMSPropOptimizer(learning_rate),
        'AdamOptimizer':tf.train.AdamOptimizer(learning_rate),
        'FtrlOptimizer':tf.train.FtrlOptimizer(learning_rate),
    }

    loss_dict={}
    acc_dict={}
    for key in optimizeres:
        loss_dict[key]=[]
        net = network(optimizeres[key])
        print(key)
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
                    loss_dict[key].append(loss_totle)
                print(epoch, "cross_entropy:", loss_dict[key][-1])
            acc_dict[key]= net.accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print("accuracy", acc_dict[key])       

    plt.figure()
    c = iter(cm.rainbow(np.linspace(0, 1, len(optimizeres))))
    for key in optimizeres:
        color = next(c)
        y = loss_dict[key]
        x = np.linspace(0, len(y), len(y))
        plt.plot(x, y, label='%s (acc: %s) loss'%(key, acc_dict[key]), color=color)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

