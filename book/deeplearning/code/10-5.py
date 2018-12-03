# RNN 固定长度的x序列映射到定长的y
# 输入字母序列，输出这个序列中定长小写字母y：

import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

class dataset():
    def __init__(self):
        # 用错位的字符串序列表示有规律的序列样本
        self.chars='AzBaCbDcEdFeGfHgIhJiKjLkMlNmOnPoQpRqSrTsUtVuWvXwYxZy'
        self.chars_length=len(self.chars)
    
    def next_batch(self, batch_size, x_seq_len, y_seq_len): 
        train_x = np.zeros([batch_size, x_seq_len, self.chars_length])
        train_y = np.zeros([batch_size, y_seq_len, self.chars_length])
        for i in range(batch_size):
            # 从 chars 中随机截取长度为 seq_len 的字符串
            idx = random.randint(0,self.chars_length-1-x_seq_len)
            _y_seq_len = 0
            for j in range(x_seq_len):
                c = self.chars[idx+j]
                train_x[i][j][idx+j]=1
                if str.islower(c) and _y_seq_len<y_seq_len:
                    train_y[i][_y_seq_len][idx+j]=1
                    _y_seq_len += 1
        return train_x, train_y

ds=dataset()
class network():
    def add_rnn_layer(self, inputs, x_seq_len, y_seq_len, batch_size, cell_num ):
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=cell_num, activation=tf.nn.relu)
        

        R = tf.Variable(tf.random_normal([x_seq_len*ds.chars_length, ds.chars_length]))
        x_r = tf.matmul(inputs, R)
        x_r = tf.reshape(x_r,[-1, ds.chars_length])

        # 将x的R的矩阵作为初始，同时也作为整个序列给cell
        init_state = x_r

        y = tf.transpose(self.y,[1,0,2])
        def compute(i, cur_state, out):
            output, next_state = cell(x_r, cur_state)

            # 训练时直接将当前的样本标签作为下一个输入的状态
            # 预测时用本身的输出作为下一个输入的状态
            next_state = tf.cond(self.training, lambda: y[i], lambda: output)
            return i+1, next_state, out.write(i, output)

        time = y_seq_len

        # 按时间轴循环计算
        _, cur_state, out = tf.while_loop(
            lambda a, *_: a < time,
            compute,
            (0, init_state, tf.TensorArray(tf.float32, time))
        )

        outputs = tf.transpose(out.stack(),[1,0,2])

        return outputs, cur_state

    def __init__(self, x_seq_len, y_seq_len):
        self.x = tf.placeholder(tf.float32, [None, x_seq_len, ds.chars_length], name='x')
        self.y = tf.placeholder(tf.float32, [None, y_seq_len, ds.chars_length], name='y')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.training = tf.placeholder_with_default(True, shape=(), name='training')

        # 将x转为定长序列
        inputs = tf.reshape(self.x, [-1, x_seq_len*ds.chars_length])
        self.pred, _ =  self.add_rnn_layer(inputs, x_seq_len, y_seq_len, self.batch_size, ds.chars_length)

        # 这里并没有使用 softmax，而是直接采用 sigmoid 计算所有的概率，这样有助于网络的稳定。
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred,1),tf.argmax(self.y,1)),tf.float32)) 

def main():
    loss_list=[]
    x_seq_len=15
    y_seq_len=5
    net = network(x_seq_len, y_seq_len)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 500
        loss_totle = 0 
        for epoch in range(10000):
            batch_xs, batch_ys = ds.next_batch(batch_size, x_seq_len, y_seq_len)
            loss,_= sess.run([net.cross_entropy, net.optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys, net.batch_size: batch_size})
            if loss_totle==0:
                loss_totle=loss
            else:
                loss_totle=loss_totle*0.99+0.01*loss
            loss_list.append(loss_totle)
            if epoch % 100 == 0:
                test_xs, test_ys = ds.next_batch(batch_size, x_seq_len, y_seq_len)
                acc = net.accuracy.eval({net.x: test_xs, net.y: test_ys, net.batch_size: batch_size, net.training: False})
                print(epoch, "cross_entropy:", loss_list[-1],"acc:", acc)

    plt.figure()
    x = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x, loss_list, label='acc: %s loss'%acc, color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()