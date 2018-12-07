# 简单 CNN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

    # 增加lstm
    def add_lstm_layer(self, inputs, num_units):
        lstm_cell =  tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1)
        outputs, _ = tf.nn.dynamic_rnn(
            cell=lstm_cell,              # 选择传入的cell
            inputs=inputs,               # 传入的数据
            initial_state=None,         # 初始状态
            dtype=tf.float32,           # 数据类型
            time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)，这里根据image结构选择False
        )
        return outputs[:,-1,:]          # 取最后一个输出用

    # 增加GRU加双向RNN
    def add_bi_gru_layer(self, inputs, num_units):
        gru_fw_cell = tf.nn.rnn_cell.GRUCell(num_units)
        gru_bw_cell = tf.nn.rnn_cell.GRUCell(num_units)
        outputs, _  = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, inputs, dtype=tf.float32)
        # 双向RNN 返回的 outputs 包括了2个输出，output_fw, output_bw ，直接合并取最后一个输出
        return tf.concat(outputs, axis=-1)[:,-1,:] 

    def __init__(self, lstm=True):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        # 将输入的图片变为 [batch_size, max_time, seq_len] 的形式，提交RNN网络
        x_image = tf.reshape(self.x, [-1, 28, 28])
        # 返回的是最后一个输出即： [batch_size, num_units]
        if lstm:
            layer = self.add_lstm_layer(x_image, 32)
        else: 
            layer = self.add_bi_gru_layer(x_image, 32)
        # 由于需要预测10个数字，接一个全连接层，输入10
        self.fc_layer = tf.layers.dense(layer, 10)

        self.loss = tf.losses.softmax_cross_entropy(self.y, self.fc_layer)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fc_layer,1), tf.argmax(self.y,1)), tf.float32))
        self.optimizer= tf.train.AdamOptimizer(0.001).minimize(self.loss)
    
def main():
    loss_list=[]
    # 使用LSTM
    net = network(True)
    # 或使用双向GRU
    # net = network(False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 100
        loss_totle = 0 
        for epoch in range(10):
            total_batch = int(mnist.train.num_examples / batch_size)
            for step in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                loss,_= sess.run([net.loss,net.optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})
                if loss_totle==0:
                    loss_totle=loss
                else:
                    loss_totle=loss_totle*0.99+0.01*loss
                loss_list.append(loss_totle)
            acc = net.accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print(epoch, "cross_entropy:", loss_list[-1],"acc:", acc)

    plt.figure()
    x = np.linspace(0, len(loss_list), len(loss_list))
    plt.plot(x, loss_list, label='acc: %s loss'%acc, color='r')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

