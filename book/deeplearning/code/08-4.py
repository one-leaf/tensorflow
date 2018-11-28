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
        # 增加 batch_normalization
    def batch_norm(self, input, eps=1e-05):
        ema = tf.train.ExponentialMovingAverage(decay=0.5)  
        input_shape = input.get_shape().as_list() 
        axes = list(range(len(input_shape) - 1))

        pop_mean, pop_var = tf.nn.moments(input, axes)
        def mean_var_with_update():
            ema_apply_op = ema.apply([pop_mean, pop_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(pop_mean), tf.identity(pop_var)
        mean, variance = tf.cond(self.training, mean_var_with_update, lambda: (ema.average(pop_mean), ema.average(pop_var)))
        beta = tf.Variable(initial_value=tf.zeros(input_shape[-1]))
        gamma = tf.Variable(initial_value=tf.ones(input_shape[-1]))
        output  = tf.nn.batch_normalization(input, mean, variance, beta, gamma, eps)
        return output 

    # 增加层
    def add_layer(self, input, out_size, active_fun, bn):
        _, in_size = input.get_shape().as_list() 
        Weights = tf.Variable(tf.random_uniform([in_size, out_size],-1,1))
        biases = tf.Variable(tf.zeros([out_size]) + 0.1)
        Wx_plus_b = tf.matmul(input, Weights) + biases
        if bn:
            Wx_plus_b = self.batch_norm(Wx_plus_b)
        if active_fun:
            return tf.nn.relu(Wx_plus_b)
        return Wx_plus_b

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        self.training = tf.placeholder_with_default(True, shape=(), name='training')

        # 教师网络
        self.teacher_layers = [self.x]
        layer_widths=[512,128,32,128,512,10]
        for width in layer_widths:
            if width == layer_widths[-1]:
                layer = self.add_layer(self.teacher_layers[-1], width, False, False)
            else:
                layer = self.add_layer(self.teacher_layers[-1], width, True, True)
            self.teacher_layers.append(layer)
        self.teacher_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.teacher_layers[-1]))
        self.teacher_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.teacher_layers[-1],1),tf.argmax(self.y,1)),tf.float32))

        # 学生网络
        self.student_layers = [self.x]
        layer_widths=[32,32,32,32,32,32,32,32,32,32,32,10]
        for width in layer_widths:
            with tf.variable_scope('student_network'):
                if width==layer_widths[-1]:
                    layer = self.add_layer(self.student_layers[-1], width, False, False)
                else:
                    layer = self.add_layer(self.student_layers[-1], width, True, True)
                self.student_layers.append(layer)
        self.student_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.student_layers[-1]))
        self.student_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.student_layers[-1],1),tf.argmax(self.y,1)),tf.float32))
        student_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='student_network')

        # 中间层损失
        self.teacher_student_loss = tf.losses.mean_squared_error(self.teacher_layers[3], self.student_layers[6])
 
        self.teacher_optimizer= tf.train.GradientDescentOptimizer(0.001).minimize(self.teacher_cross_entropy+self.teacher_student_loss)
        self.student_optimizer= tf.train.GradientDescentOptimizer(0.001).minimize(self.student_cross_entropy, var_list=student_vars)       

def main():
    loss_dict={}
    acc_dict={}

    loss_dict["teacher"]=[]
    loss_dict["student"]=[]
    loss_dict["teacher-student"]=[]

    for has_teacher in [True, False]:
        net = network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 100

            if has_teacher:
                epochs=[15,5]
            else:
                epochs=[0,20]

            # 先训练教师网络和将学生的中层网络和教师的中层一致
            for epoch in range(epochs[0]):
                total_batch = int(mnist.train.num_examples / batch_size)
                for step in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    loss_t,loss_ts,_= sess.run([net.teacher_cross_entropy, net.teacher_student_loss, net.teacher_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})   
                acc_t = net.teacher_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels, net.training: False})
                print(epoch,'teacher loss:' ,loss_t, 'teacher_student loss:' ,loss_ts, 'teacher acc:', acc_t)
                if loss_ts<0.003 and acc_t>0.97: break

            # 训练学生网络
            for epoch in range(epochs[1]):
                total_batch = int(mnist.train.num_examples / batch_size)
                for step in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    loss_s,_= sess.run([net.student_cross_entropy, net.student_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})       
                acc_s = net.student_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels, net.training: False})
                print(epoch, 'student loss:', loss_s, 'student acc:', acc_s)

if __name__ == '__main__':
    main()

