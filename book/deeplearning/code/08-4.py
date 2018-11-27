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
        layer_widths=[32,32,32,32,32,16,16,16,16,10]
        for i, width in enumerate(layer_widths):
            if width==layer_widths[-1]:
                layer = self.add_layer(self.student_layers[-1], width, False, False, 'student_layer_%s'%i)
            else:
                layer = self.add_layer(self.student_layers[-1], width, True, False, 'student_layer_%s'%i)
            self.student_layers.append(layer)
        self.student_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.student_layers[-1]))
        self.student_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.student_layers[-1],1),tf.argmax(self.y,1)),tf.float32))

        # 中间层学习
        self.teacher_student_loss = tf.losses.mean_squared_error(self.teacher_layers[3], self.student_layers[5])

        self.student_optimizer= tf.train.AdamOptimizer(0.000001).minimize(self.student_cross_entropy)
        
        self.teacher_optimizer= tf.train.AdamOptimizer(0.01).minimize(self.teacher_cross_entropy+self.teacher_student_loss)

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
         
        # 先训练教师网络
        print("Start train teacher network ...")
        for epoch in range(50):
            total_batch = int(mnist.train.num_examples / batch_size)
            for step in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                loss_t,_= sess.run([net.teacher_cross_entropy, net.teacher_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})
            acc = net.teacher_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print(epoch,'loss:' ,loss_t, 'acc:', acc)

        # # 再训练学生网络
        # print("Start train student middle network ...")
        # # for epoch in range(50):
        # while True:
        #     for step in range(total_batch):
        #         loss_s,_= sess.run([net.teacher_student_loss, net.teacher_student_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})
        #     print(epoch, 'loss:', loss_s, )
        #     if loss_s<1: break

        print("Start train student end network ...")
        for epoch in range(50):
            for step in range(total_batch):
                loss_s,_= sess.run([net.student_cross_entropy, net.student_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})       
            acc = net.student_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print(epoch, 'loss:', loss_s, 'acc:', acc)
            # acc_dict["student"]= 
            # print("accuracy", acc_dict["student"])       

    # plt.figure()
    # c = iter(cm.rainbow(np.linspace(0, 1, 2)))
    # for normal in [True, False]:
    #     color = next(c)
    #     y = loss_dict[normal]
    #     x = np.linspace(0, len(y), len(y))
    #     plt.plot(x, y, label='BN: %s (acc: %s) loss'%(normal, acc_dict[normal]), color=color)
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    main()

