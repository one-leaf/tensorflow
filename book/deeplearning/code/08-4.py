# 测试批标准化
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
    # 增加层
    def add_layer(self, input, out_size, active_fun, bn, name):
        _, in_size = input.get_shape().as_list() 
        Weights = tf.Variable(tf.random_uniform([in_size, out_size],-1,1))
        biases = tf.Variable(tf.zeros([out_size]) + 0.1)
        Wx_plus_b = tf.matmul(input, Weights) + biases
        if bn:
            Wx_plus_b = tf.layers.batch_normalization(Wx_plus_b, training=True, name='%s_bn'%name)
        if active_fun:
            return tf.nn.relu(Wx_plus_b)
        return Wx_plus_b

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        # 教师网络
        self.teacher_layers = [self.x]
        layer_widths=[512,128,32,128,512,10]
        for i, width in enumerate(layer_widths):
            if width == layer_widths[-1]:
                layer = self.add_layer(self.teacher_layers[-1], width, False, False, 'teacher_layer_%s'%i)
            else:
                layer = self.add_layer(self.teacher_layers[-1], width, True, True, 'teacher_layer_%s'%i)
            self.teacher_layers.append(layer)
        self.teacher_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.teacher_layers[-1]))
        self.teacher_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.teacher_layers[-1],1),tf.argmax(self.y,1)),tf.float32))

        # 学生网络
        self.student_layers = [self.x]
        layer_widths=[32,32,32,32,32,32,32,32,32,32,32,10]
        for i, width in enumerate(layer_widths):
            with tf.variable_scope('student_network'):
                if width==layer_widths[-1]:
                    layer = self.add_layer(self.student_layers[-1], width, False, False, 'student_layer_%s'%i)
                else:
                    layer = self.add_layer(self.student_layers[-1], width, True, True, 'student_layer_%s'%i)
                self.student_layers.append(layer)
        self.student_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.student_layers[-1]))
        self.student_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.student_layers[-1],1),tf.argmax(self.y,1)),tf.float32))
        student_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='student_network')

        # 中间层损失
        self.teacher_student_loss = tf.losses.mean_squared_error(self.teacher_layers[3], self.student_layers[6])
        self.teacher_student_optimizer= tf.train.AdamOptimizer(0.001).minimize(self.teacher_student_loss, var_list=student_vars)

        self.teacher_optimizer= tf.train.AdamOptimizer(0.001).minimize(self.teacher_cross_entropy+self.teacher_student_loss)
        self.student_optimizer= tf.train.AdamOptimizer(0.0001).minimize(self.student_cross_entropy, var_list=student_vars)       

def main():
    loss_dict={}
    acc_dict={}

    loss_dict["teacher"]=[]
    loss_dict["student"]=[]
    loss_dict["teacher-student"]=[]
    net = network()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 100

        # 先训练教师网络和将学生的中层网络和教师的中层一致
        # 如果需要看效果，可以屏蔽此部分，直接训练学生网络 
        for epoch in range(19):
            total_batch = int(mnist.train.num_examples / batch_size)
            for step in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                loss_t,loss_ts,_= sess.run([net.teacher_cross_entropy, net.teacher_student_loss, net.teacher_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})   
            acc_t = net.teacher_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print(epoch,'teacher loss:' ,loss_t, 'teacher_student loss:' ,loss_ts, 'teacher acc:', acc_t)
            if loss_ts<0.003 and acc_t>0.97: break

        # 训练学生网络
        for epoch in range(1):
            total_batch = int(mnist.train.num_examples / batch_size)
            for step in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                loss_s,_= sess.run([net.student_cross_entropy, net.student_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})       
            acc_s = net.student_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print(epoch, 'student loss:', loss_s, 'student acc:', acc_s)

    net = network()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 100

        # 训练学生网络
        for epoch in range(20):
            total_batch = int(mnist.train.num_examples / batch_size)
            for step in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                loss_s,_= sess.run([net.student_cross_entropy, net.student_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})       
            acc_s = net.student_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print(epoch, 'student loss:', loss_s, 'student acc:', acc_s)


if __name__ == '__main__':
    main()

