# RNN模式2
# 输入字母序列，全部转小写：

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

class dataset():
    def __init__(self):
        # 用错位的字符串序列表示有规律的序列样本
        self.chars='AzBaCbDcEdFeGfHgIhJiKjLkMlNmOnPoQpRqSrTsUtVuWvXwYxZy'
        self.chars_length=len(self.chars)
    
    def next_batch(self, batch_size, seq_len): 
        train_x = np.zeros([batch_size, seq_len, self.chars_length])
        train_y = np.zeros([batch_size, seq_len, self.chars_length])
        for i in range(batch_size):
            # 从 chars 中随机截取长度为 seq_len 的字符串
            idx = random.randint(0,self.chars_length-1-seq_len)
            for j in range(seq_len):
                c = self.chars[idx+j]
                lower_c = str.lower(c)
                train_x[i][j][idx+j]=1
                train_y[i][j][self.chars.index(lower_c)]=1
        return train_x, train_y

ds=dataset()
class network():
    def add_rnn_layer(self, inputs, batch_size, cell_num ):
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=cell_num, activation=tf.nn.tanh)
        init_state = cell.zero_state(batch_size, np.float32)

        # 将 时间轴移到第一个，方便计算
        inputs = tf.transpose(inputs,[1,0,2])

        y = tf.transpose(self.y,[1,0,2])
        def compute(i, cur_state, out):
            output, _ = cell(inputs[i], cur_state)

            # 直接将当前输出和标签做一次对数作为下一个输入的状态
            # 注意，这里并没有考虑预测时y的输入，实际预测时将output的输出作为下一次的输入状态
            output = -y[i]*tf.log(tf.maximum(output,1e-9))
            cur_state = output
            return i+1, cur_state, out.write(i, output)

        time = tf.shape(inputs)[0]

        # 按时间轴循环计算
        _, cur_state, out = tf.while_loop(
            lambda a, *_: a < time,
            compute,
            (0, init_state, tf.TensorArray(tf.float32, time))
        )

        outputs = tf.transpose(out.stack(),[1,0,2])

        return outputs, cur_state

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, None, ds.chars_length], name='x')
        self.y = tf.placeholder(tf.float32, [None, None, ds.chars_length], name='y')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        self.pred, _ =  self.add_rnn_layer(self.x, self.batch_size, ds.chars_length)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred,1),tf.argmax(self.y,1)),tf.float32)) 

def main():
    loss_list=[]
    net = network()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 500
        loss_totle = 0 
        for epoch in range(2000):
            seq_len = random.randint(5,10)
            batch_xs, batch_ys = ds.next_batch(batch_size, seq_len)
            loss,_= sess.run([net.cross_entropy,net.optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys, net.batch_size: batch_size})
            if loss_totle==0:
                loss_totle=loss
            else:
                loss_totle=loss_totle*0.99+0.01*loss
            loss_list.append(loss_totle)
            if epoch % 10 == 0:
                test_xs, test_ys = ds.next_batch(batch_size, seq_len)
                acc = net.accuracy.eval({net.x: test_xs, net.y: test_ys, net.batch_size: batch_size})
                print(epoch, "cross_entropy:", loss_list[-1],"acc:", acc)

    plt.figure()
    x = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x, loss_list, label='acc: %s loss'%acc, color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()