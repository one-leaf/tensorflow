# 测试常见优化算法
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
    def __init__(self, loss_optimizer):
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')
        h_w = tf.Variable(tf.random_uniform([784, 784],-1,1))
        h_b = tf.Variable(tf.zeros([784]))
        hide_layer = tf.nn.relu(tf.add(tf.matmul(self.x, h_w), h_b))
        w = tf.Variable(tf.random_uniform([784, 10],-1,1))
        b = tf.Variable(tf.zeros([10]))
        self.full_connect_layer = tf.add(tf.matmul(hide_layer, w), b)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.full_connect_layer))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.full_connect_layer,1),tf.argmax(self.y,1)),tf.float32))
        self.optimizer= loss_optimizer.minimize(self.cross_entropy)
    
def main():
    learning_rate = 0.001
    optimizeres = {
        'GradientDescentOptimizer':tf.train.GradientDescentOptimizer(learning_rate),
        'AdagradOptimizer':tf.train.AdagradOptimizer(learning_rate),
        'MomentumOptimizer':tf.train.MomentumOptimizer(learning_rate, momentum=0.9),
        'RMSPropOptimizer':tf.train.RMSPropOptimizer(learning_rate),
        'AdamOptimizer':tf.train.AdamOptimizer(learning_rate),
        'FtrlOptimizer':tf.train.FtrlOptimizer(learning_rate),
    }

    loss_dict={}
    acc_dict={}
    for key in optimizeres:
        loss_dict[key]=[]
        net = network(optimizeres[key])
        print(key)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            batch_size = 100
            loss_totle = 0 
            for epoch in range(10):
                total_batch = int(mnist.train.num_examples / batch_size)
                for step in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    loss,_= sess.run([net.cross_entropy,net.optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})
                    if loss_totle==0:
                        loss_totle=loss
                    else:
                        loss_totle=loss_totle*0.99+0.01*loss
                    loss_dict[key].append(loss_totle)
                print(epoch, "cross_entropy:", loss_dict[key][-1])
            acc_dict[key]= net.accuracy.eval({net.x: mnist.test.images, net.y: mnist.test.labels})
            print("accuracy", acc_dict[key])       

    plt.figure()
    c = iter(cm.rainbow(np.linspace(0, 1, len(optimizeres))))
    for key in optimizeres:
        color = next(c)
        y = loss_dict[key]
        x = np.linspace(0, len(y), len(y))
        plt.plot(x, y, label='%s (acc: %s) loss'%(key, acc_dict[key]), color=color)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

