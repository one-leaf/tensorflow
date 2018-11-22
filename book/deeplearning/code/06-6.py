# 反向传播1
import numpy as np
import tensorflow as tf
import os

# 导入手写体数据
from tensorflow.examples.tutorials.mnist import input_data
curr_path = os.path.dirname(os.path.realpath(__file__))
mnist = input_data.read_data_sets(os.path.join(curr_path,"../data"), one_hot=True)

# 定义神经网络
class network():
    def __init__(self):
        self.learning_rate = 0.001
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.w = tf.Variable(tf.random_uniform([784, 10],-1,1), name="weights")
        self.b = tf.Variable(tf.zeros([10]), name="bias")
        self.full_connect_layer = tf.add(tf.matmul(self.x, self.w), self.b)
        self.pred = tf.nn.softmax(self.full_connect_layer,name='y_pred')

    # 获得正确率
    def get_accuracy(self):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred,1),tf.argmax(self.y,1)),tf.float32))
        return accuracy

    # 自己算梯度更新
    def get_loss1(self):
        # 通过设置log前的最小值不让归0，防止出现 log(0) 未定义
        tf.clip_by_value(self.pred, 1e-15, 1.0)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1)) 

        w_grad =  - tf.matmul ( tf.transpose(self.x) , self.y - self.pred) 
        b_grad = - tf.reduce_mean( tf.matmul(tf.transpose(self.x), self.y - self.pred), reduction_indices=0)

        new_w = self.w.assign(self.w - self.learning_rate * w_grad)
        new_b = self.b.assign(self.b - self.learning_rate * b_grad)
        optimizer=[new_w, new_b]
        return cross_entropy, optimizer

    # tf算梯度更新
    def get_loss2(self):
        # 通过设置log前的最小值不让归0，防止出现 log(0) 未定义
        tf.clip_by_value(self.pred, 1e-15, 1.0)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1)) 

        w_grad, b_grad=tf.gradients(cross_entropy,[self.w,self.b])

        new_w = self.w.assign(self.w - self.learning_rate * w_grad)
        new_b = self.b.assign(self.b - self.learning_rate * b_grad)
        optimizer=[new_w, new_b]
        return cross_entropy, optimizer

    # tf随机梯度下降
    def get_loss3(self):
        # 通过设置log前的最小值不让归0，防止出现 log(0) 未定义
        tf.clip_by_value(self.pred, 1e-15, 1.0)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.pred), reduction_indices=1)) 
        optimizer= tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)
        return cross_entropy, optimizer

def main():
    net = network()
    cross_entropy, optimizer = net.get_loss1()
    accuracy = net.get_accuracy()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    for epoch in range(10):
        total_batch = int(mnist.train.num_examples / batch_size)
        for step in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

        print("cross_entropy:", cross_entropy.eval({x: batch_xs, y: batch_ys}))
    print('accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
if __name__ == '__main__':
    main()

