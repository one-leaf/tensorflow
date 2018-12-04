# RNN 回声网络

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

seq_len = 15
state_size = 4
num_classes = 2
batch_size = 5
total_series_length = batch_size*seq_len*1000

# x 从0和1中间 随机选择长度为 total_series_length
# y 将x的数据往前移动 echo_step 位，x最后位会移到y前面
def dateset(echo_step = 3):
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    x = x.reshape((batch_size, -1)) 
    y = y.reshape((batch_size, -1))

    for i in range(total_series_length//batch_size//seq_len):
        start_idx = i*seq_len 
        end_idx = start_idx + seq_len
        yield x[:,start_idx:end_idx],y[:,start_idx:end_idx] 

class network():
    def add_esn_layer(self, inputs, init_state):
        W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
        b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)
    
        current_state = init_state
        states = []
        inputs = tf.unstack(inputs, axis=1)
        for input in inputs:
            input = tf.reshape(input, [batch_size, 1])
            input_and_state_concatenated = tf.concat([input, current_state], 1)  
            next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b) 
            states.append(next_state)
            current_state = next_state

        return states

    def __init__(self):
        self.x = tf.placeholder(tf.float32, [batch_size, seq_len])
        self.y = tf.placeholder(tf.float32, [batch_size, seq_len])
        self.init_state = tf.placeholder(tf.float32, [batch_size, state_size])

        states = self.add_esn_layer(self.x, self.init_state)
        self.current_state = states[-1]
        
        layer = tf.stack(states, 1) #[batch_size, seq_len, state_size]
        W2 = tf.Variable(np.random.rand(batch_size, state_size, 1),dtype=tf.float32)
        b2 = tf.Variable(np.random.rand(batch_size, seq_len, 1),dtype=tf.float32)
        layer = tf.matmul(layer, W2) + b2 #[batch_size, seq_len, 1]
        logits = tf.reshape(layer,[batch_size, seq_len])
        self.pred = tf.nn.sigmoid(logits)
 
        self.losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=logits))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.losses)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.pred),tf.round(self.y)),tf.float32))

def plot(loss_list, pred, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    for batch_series_idx in range(batch_size):
        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, seq_len, 0, 1])
        left_offset = range(seq_len)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.8, width=1, color="red")
        plt.bar(left_offset, pred[batch_series_idx, :] * 0.4, width=1, color="green")
    plt.draw()
    plt.pause(0.0001)

net = network()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []    
    
    for epoch_idx in range(10):      
        _current_state = np.zeros((batch_size, state_size))
        ds = dateset()
        for batch_idx, (batchX, batchY) in enumerate(ds):
            _losses, _train_step, _current_state, _pred, _acc = sess.run(
                [net.losses, net.train_step, net.current_state, net.pred, net.accuracy],
                feed_dict={
                    net.x:batchX,
                    net.y:batchY,
                    net.init_state: _current_state
                })
            loss_list.append(_losses)            
            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _losses, "Acc", _acc)
                plot(loss_list, _pred, batchX, batchY)

plt.ioff()
plt.show()