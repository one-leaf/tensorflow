# 测试批标准化
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
            if width==layer_widths[-1]:
                layer = self.add_layer(self.student_layers[-1], width, False, False)
            else:
                layer = self.add_layer(self.student_layers[-1], width, True, True)
            self.student_layers.append(layer)
        self.student_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.student_layers[-1]))
        self.student_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.student_layers[-1],1),tf.argmax(self.y,1)),tf.float32))

        # 中间层损失
        self.teacher_student_loss = tf.losses.mean_squared_error(self.teacher_layers[3], self.student_layers[6])
 
        self.teacher_optimizer= tf.train.AdamOptimizer(0.01).minimize(self.teacher_cross_entropy+self.teacher_student_loss)
        self.student_optimizer= tf.train.GradientDescentOptimizer(0.01).minimize(self.student_cross_entropy)       

def main():
    acc_dict={}
    # 总共训练次数
    epochs_count = 10
    acc_dict["teacher-student"]=[]
    acc_dict["student"]=[]

    for has_teacher in [True, False]:
        net = network()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 100

            if has_teacher:
                epochs=[epochs_count-5,5]
                acc_list = acc_dict["teacher-student"]
            else:
                epochs=[0,epochs_count]
                acc_list = acc_dict["student"]

            # 先训练教师网络和将学生的中层网络和教师的中层一致
            for epoch in range(epochs[0]):
                total_batch = int(mnist.train.num_examples / batch_size)
                for step in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    loss_t,loss_ts,_= sess.run([net.teacher_cross_entropy, net.teacher_student_loss, net.teacher_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})   
                acc_t = net.teacher_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels, net.training: False})
                print(epoch,'teacher loss:' ,loss_t, 'teacher_student loss:' ,loss_ts, 'teacher acc:', acc_t)
                if loss_ts<0.003 and acc_t>0.97: break
                acc_list.append(0)

            # 训练学生网络
            for epoch in range(epochs[1]):
                total_batch = int(mnist.train.num_examples / batch_size)
                for step in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    loss_s,_= sess.run([net.student_cross_entropy, net.student_optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})       
                acc_s = net.student_accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels, net.training: False})
                print(epoch, 'student loss:', loss_s, 'student acc:', acc_s)
                acc_list.append(acc_s)

    plt.figure()
    x = np.linspace(0, epochs_count, epochs_count)
    plt.plot(x, acc_dict["teacher-student"], label='T-S acc', color='r')
    plt.plot(x, acc_dict["student"], label='S acc', color='b')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

