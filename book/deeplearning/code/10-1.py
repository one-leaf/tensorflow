# RNN模式1
# 输入字母序列，全部转小写：

import numpy as np
import tensorflow as tf
import random

class dataset():
    def __init__(self):
        self.chars='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
        self.chars_length=len(chars)
    
    def next_batch(self, batch_size, seq_len): 
        train_x = np.zeros(batch_size, seq_len, self.chars_length)
        train_y = np.zeros(batch_size, seq_len, self.chars_length)
        for i in range(batch_size):
            for j in range(seq_len):
                c = random.choice(self.chars)
                lower_c = str.lower(c)
                train_x[i][j][self.chars.index(c)]=1
                train_y[i][j][self.chars.index(lower_c)]=1
        return train_x, train_y

ds=dataset()
class network():
    def add_rnn_layer(self, inputs, batch_size, cell_num, seq_len):
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=cell_num)
        state = cell.zero_state(batch_size, np.float32)

        # 将 时间轴移到第一个，方便计算
        # 按时间轴循环取数据进行 rnn
        inouts = tf.transpose(inputs,[1,0,2])
        outputs=[]
        for i in range(seq_len):
            output, state = cell.call(inputs[i], state)
            outputs.append(output)

        # 合并所有的序列
        output = tf.concat(outputs)
        # 将时间轴移动回第二个
        output = tf.transpose(output,[1,0,2])

        return output, state

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, None, ds.chars_length], name='x')
        self.y = tf.placeholder(tf.float32, [None, None, ds.chars_length], name='y') self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        self.pred, _ =  add_rnn_layer(self.x, self.batch_size, 8)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred,1),tf.argmax(self.y,1)),tf.float32)) 


def main():
    loss_list=[]
    net = network()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 50
        loss_totle = 0 
        for epoch in range(10):
            total_batch = 100
            for _ in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                loss,_,step= sess.run([net.cross_entropy,net.optimizer,net.global_step], feed_dict={net.x: batch_xs, net.y: batch_ys})
                if loss_totle==0:
                    loss_totle=loss
                else:
                    loss_totle=loss_totle*0.99+0.01*loss
                loss_list.append(loss_totle)
                if step % 10 == 0:
                    test_xs, test_ys = mnist.test.next_batch(batch_size)
                    acc = net.accuracy.eval({net.x: test_xs, net.y: test_ys, net.training: False})
                    print(epoch, '/' ,step, "cross_entropy:", loss_list[-1],"acc:", acc)

    plt.figure()
    x = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x, loss_list, label='acc: %s loss'%acc, color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()