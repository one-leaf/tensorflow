# coding=utf-8
# sudo -H pip3 install pandas sklearn scipy
import requests,zipfile,io
import os
import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn import preprocessing

curr_dir = os.path.dirname(__file__)
model_dir = os.path.join(curr_dir, "model")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
data_dir = os.path.join(curr_dir, "data")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

trainingDataFile = os.path.join(data_dir,'UJIndoorLoc','trainingData.csv')
validationDataFile = os.path.join(data_dir,'UJIndoorLoc','validationData.csv')

# 下载数据
if not os.path.exists(trainingDataFile):
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip'
    request = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(request.content))
    z.extractall(data_dir)

# 数据整理
def cleanData(data):
    # 取前520个列的值
    x = np.asarray(data.ix[:,0:520])
    # 将数据按均值为0，方差为1，进行标准化
    x = preprocessing.scale(x)
    # 取出数据中的建筑和楼层两个栏位并作为字符串合并为一个2位长度的字符串
    y = np.asarray(data["BUILDINGID"].map(str) + data["FLOOR"].map(str))
    # 对结果数据进行one-hot化
    y = np.asarray(pd.get_dummies(y))
    return x,y

training_data = pd.read_csv(trainingDataFile,header = 0)
train_x, train_y = cleanData(training_data)

test_data = pd.read_csv(validationDataFile,header = 0)
test_x, test_y = cleanData(test_data)

# 输出的长度
output = train_y.shape[1]
print(train_x.shape,train_y.shape)

X = tf.placeholder(tf.float32, shape=[None, 520])  # 网络输入
Y = tf.placeholder(tf.float32,[None, output]) # 网络输出

# 定义神经网络
def neural_networks():
    # --------------------- Encoder -------------------- #
    e_w_1 = tf.Variable(tf.truncated_normal([520, 256], stddev = 0.1))
    e_b_1 = tf.Variable(tf.constant(0.0, shape=[256]))
    e_w_2 = tf.Variable(tf.truncated_normal([256, 128], stddev = 0.1))
    e_b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    e_w_3 = tf.Variable(tf.truncated_normal([128, 64], stddev = 0.1))
    e_b_3 = tf.Variable(tf.constant(0.0, shape=[64]))
    # --------------------- Decoder  ------------------- #
    d_w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev = 0.1))
    d_b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
    d_w_2 = tf.Variable(tf.truncated_normal([128, 256], stddev = 0.1))
    d_b_2 = tf.Variable(tf.constant(0.0, shape=[256]))
    d_w_3 = tf.Variable(tf.truncated_normal([256, 520], stddev = 0.1))
    d_b_3 = tf.Variable(tf.constant(0.0, shape=[520]))
    # --------------------- DNN  ------------------- #
    w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev = 0.1))
    b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_2 = tf.Variable(tf.truncated_normal([128, 128], stddev = 0.1))
    b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_3 = tf.Variable(tf.truncated_normal([128, output], stddev = 0.1))
    b_3 = tf.Variable(tf.constant(0.0, shape=[output]))
    #########################################################
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(X,       e_w_1), e_b_1))    
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, e_w_2), e_b_2))
    encoded = tf.nn.tanh(tf.add(tf.matmul(layer_2, e_w_3), e_b_3))

    layer_4 = tf.nn.tanh(tf.add(tf.matmul(encoded, d_w_1), d_b_1))
    layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, d_w_2), d_b_2))
    decoded = tf.nn.tanh(tf.add(tf.matmul(layer_5, d_w_3), d_b_3))

    layer_7 = tf.nn.tanh(tf.add(tf.matmul(encoded, w_1),   b_1))
    layer_8 = tf.nn.tanh(tf.add(tf.matmul(layer_7, w_2),   b_2))
    out = tf.nn.softmax(tf.add(tf.matmul( layer_8, w_3),   b_3))
    print(layer_1.shape,layer_2.shape,encoded.shape)
    print(layer_4.shape,layer_5.shape,decoded.shape)
    print(layer_7.shape,layer_8.shape,out.shape)
    return (decoded, out)

# 训练神经网络
def train_neural_networks():
    decoded, predict_output = neural_networks()    

    us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
    us_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(us_cost_function)
    
    s_cost_function = -tf.reduce_sum(Y * tf.log(predict_output))
    s_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(s_cost_function)

    correct_prediction = tf.equal(tf.argmax(predict_output, 1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    training_epochs = 20
    batch_size = 10
    total_batches = training_data.shape[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        # 为啥需要这个逻辑，输入等于输出，找出输入数据之间的规律？
        # ------------ Training Autoencoders - Unsupervised Learning ----------- #
        # autoencoder是一种非监督学习算法，他利用反向传播算法，让目标值等于输入值
        # for epoch in range(training_epochs):
        #     epoch_costs = np.empty(0)
        #     for b in range(total_batches):
        #         offset = (b * batch_size) % (train_x.shape[0] - batch_size)
        #         batch_x = train_x[offset:(offset + batch_size), :]
        #         _, c = sess.run([us_optimizer, us_cost_function],feed_dict={X: batch_x})
        #         epoch_costs = np.append(epoch_costs, c)
        #     print("Epoch: ",epoch," Loss: ",np.mean(epoch_costs))
        # print("------------------------------------------------------------------")

        # ---------------- Training NN - Supervised Learning ------------------ #
        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                batch_y = train_y[offset:(offset + batch_size), :]
                # 加了反向传播后，正确率提供一点
                _, c = sess.run([us_optimizer, us_cost_function],feed_dict={X: batch_x})
                _, c = sess.run([s_optimizer, s_cost_function],feed_dict={X: batch_x, Y : batch_y})
                epoch_costs = np.append(epoch_costs,c)

            accuracy_in_train_set = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
            accuracy_in_test_set = sess.run(accuracy, feed_dict={X: test_x, Y: test_y})
            print("Epoch: ",epoch," Loss: ",np.mean(epoch_costs)," Accuracy: ", accuracy_in_train_set, ' ', accuracy_in_test_set)


train_neural_networks()