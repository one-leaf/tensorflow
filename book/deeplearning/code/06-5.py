# 多层感知机梯度下降
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf

# 网络深度
L = 3
# 隐藏层的宽度
H_SIZE = 4

# 获得样本数据x, y
# x，是随机2维数据，y是分类，x到原点的距离>05，分类为1，否则为0
def getData(size=100):
    x = np.random.random((size, 2))*2-1
    y = np.power(x, 2)
    y = np.sum(y, axis=1)
    y[y > 0.5] =1.
    y[y < 1]   =0.
    return x, y[:, np.newaxis]

# 定义神经网络
def network():
    x = tf.placeholder('float', [None, 2])
    y = tf.placeholder('float', [None, 1])
    layers=[x]
    for i in range(L):
        input_shape = layers[-1].get_shape().as_list()
        w = tf.Variable(tf.random_uniform([input_shape[1], H_SIZE*(2**i)],-1,1),"hide_w%s"%i)
        b = tf.Variable(tf.zeros([H_SIZE*(2**i)]),"hide_b%s"%i)
        hide_layer = tf.add(tf.matmul(layers[-1], w), b)
        hide_layer = tf.nn.relu(hide_layer)
        layers.append(hide_layer)

    input_shape = layers[-1].get_shape().as_list()
    w = tf.Variable(tf.random_uniform([input_shape[1], 1],-1,1),"full_w%s"%i)
    b = tf.Variable(tf.zeros([1]),"full_b%s"%i)
    full_connect_layer = tf.add(tf.matmul(layers[-1], w), b)

    pred = tf.nn.sigmoid(full_connect_layer)

    grads = tf.gradients(pred, [w])  

    # tf 的 sigmoid_cross_entropy_with_logits 函数加了防止极大值和极小值，不会出现梯度消失。
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=full_connect_layer))
 
    train = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.round(pred), tf.round(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    return x,y,cross_entropy,train,accuracy,pred,grads,w,b

def main():
    x,y,cross_entropy,train,accuracy,pred,grads,w,b = network()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_size = 100
    grad_list = []
    w_list = []
    lost_list = []
    for t in range(10000):
        x_data, y_data = getData(batch_size)
        _, loss, acc = sess.run([train, cross_entropy, accuracy], feed_dict={x: x_data, y: y_data})
        if t % 100==0:
            print(t, acc, loss)
            _grads,_w,_b=(sess.run([grads,w,b], feed_dict={x: x_data, y: y_data}))
            grad_list.append(np.squeeze(_grads))
            w_list.append(np.squeeze(_w))
            lost_list.append(loss)

    x_data, y_data = getData(10000)
    pred = np.round(sess.run(pred, feed_dict={x: x_data}))

    pred[pred!=y_data]=2

    # 显示预测结果
    plt.figure(1)
    plt.scatter(x_data[:, 0], x_data[:, 1], marker='o', c=pred, s=3)

    # 显示w的梯度变化
    plt.figure(2)
    c = iter(cm.rainbow(np.linspace(0, 1, len(grad_list))))
    x = np.linspace(1, 16, 16)
    for i, grad in enumerate(grad_list):
        color = next(c)
        if i%10!=0: continue
        plt.plot(x, grad, "r", label='%s'%i, color=color)
    plt.legend()

    # 显示梯度，w，和损失函数3者之间的关系
    fig = plt.figure(3); 
    from mpl_toolkits.mplot3d import Axes3D

    ax = Axes3D(fig)
    c = cm.rainbow(np.linspace(0, 1, len(lost_list)))
    ax.set_xlabel('w value'); ax.set_ylabel('grad'); ax.set_zlabel('loss')
    ax.plot(w_list, grad_list, zs=lost_list, zdir='z', c='r', lw=1, antialiased=True)   

    plt.show()

if __name__ == '__main__':
    main()

