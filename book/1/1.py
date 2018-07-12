import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt

# 参考： https://www.oudahe.com/p/52564/

# 用两个同心圆，取数据，分为3类
r1 = 140
r2 = 80
batch_size=128
def getDatasets():
    batch_x=[]
    batch_y=[]
    for i in range(batch_size):
        x1 = (random.random()-0.5)*300 
        x2 = (random.random()-0.5)*300 
        r = (x1**2 + x2**2)**0.5 + (random.random()-0.5)*10
        y = [-1,1,-1]
        if r>r1: y=[-1,-1,1]
        if r<r2: y=[1,-1,-1]
        batch_x.append([x1,x2])
        batch_y.append(y)
    return np.array(batch_x), np.array(batch_y)

# 批量矩阵乘法
def reshape_matmul(y):
    mat = tf.transpose(y)   # (batch_size, 3) ==> (3, batch_size)
    v1 = tf.expand_dims(mat, 1)  #(3, batch_size) ==> (3, 1, batch_size)
    v2 = tf.reshape(v1,[3, batch_size, 1]) #(3,1,batch_size) ==> (3,batch_size,1)
    return tf.matmul(v2, v1) # (3,batch_size,1) * (3, 1, batch_size) ==> (3, batch_size, batch_size)

def neural_networks():
    x = tf.placeholder(tf.float32, [batch_size, 2], name='x')
    y = tf.placeholder(tf.float32, [None, 3], name='y')
    prediction_grid = tf.placeholder(tf.float32, [None, 2])

    # svm
    b = tf.Variable(tf.random_normal(shape=[batch_size, 3]))

    # Gaussian (RBF) kernel
    gamma = tf.constant(-0.005)
    dist  = tf.reduce_sum(tf.square(x), 1)  #(batch_size,)
    dist  = tf.reshape(dist, [-1,1])    #(batch_size,1)
    sq_dists  = tf.multiply(2., tf.matmul(x, tf.transpose(x)))  #(batch_size,batch_size)
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))    #(batch_size,batch_size)

    # Compute SVM Model
    first_term = tf.reduce_sum(b)   #()
    print('first_term', first_term.shape)
    b_vec_cross = tf.matmul(b, tf.transpose(b)) #(256,256)
    print('b_vec_cross', b_vec_cross.shape)
    y_target_cross = reshape_matmul(y)          #(3,256,?) 
    print('y_target_cross', y_target_cross.shape)
    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2]) #(3,)
    print('second_term', second_term.shape)
    loss = tf.reduce_mean(tf.negative(tf.subtract(first_term, second_term)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

    # 预测函数
    rA = tf.reshape(tf.reduce_sum(tf.square(x),1),[-1,1])
    rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid),1),[-1,1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x, tf.transpose(prediction_grid)))), tf.transpose(rB))
    pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))  #(?,?)
    print('pred_kernel.shape',pred_kernel.shape)

    prediction_output = tf.matmul(tf.multiply(tf.transpose(y),tf.transpose(b)), pred_kernel)
    prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y,1)), tf.float32))

    return x,y,prediction_grid,prediction,accuracy,loss,optimizer

def main():
    x,y,prediction_grid,prediction,accuracy,loss,op = neural_networks()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        loss_vec = []
        batch_accuracy = []

        for epoch in range(5000):
            train_x, train_y = getDatasets()
            _, cost, acc = sess.run([op,loss,accuracy],feed_dict={x:train_x,y:train_y,prediction_grid:train_x})
            loss_vec.append(cost)
            batch_accuracy.append(acc)
            if epoch%50==0:
                print(epoch, cost, acc)

        test_x, test_y = getDatasets()
        x_min, x_max = -160, 160
        y_min, y_max = -160, 160
        print(x_min, x_max, y_min, y_max)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
        print('xx.shape',xx.shape)
        print('yy.shape',yy.shape)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        print('grid_points.shape',grid_points.shape)
        grid_predictions = sess.run(prediction,feed_dict={x: test_x, y: test_y, prediction_grid:grid_points})
        print('grid_predictions.shape',grid_predictions.shape)
        grid_predictions = grid_predictions.reshape(xx.shape)
        print('grid_predictions.shape',grid_predictions.shape)

        # Plot points and grid
        test_y = np.argmax(test_y, axis=1)
        class1_x = [x[0] for i,x in enumerate(test_x) if test_y[i]==0]
        class1_y = [x[1] for i,x in enumerate(test_x) if test_y[i]==0]
        class2_x = [x[0] for i,x in enumerate(test_x) if test_y[i]==1]
        class2_y = [x[1] for i,x in enumerate(test_x) if test_y[i]==1]
        class3_x = [x[0] for i,x in enumerate(test_x) if test_y[i]==2]
        class3_y = [x[1] for i,x in enumerate(test_x) if test_y[i]==2]

        plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)

        plt.plot(class1_x, class1_y, 'ro', label='<80')
        plt.plot(class2_x, class2_y, 'kx', label='>80 and <140')
        plt.plot(class3_x, class3_y, 'gv', label='>140')
        plt.title('Gaussian SVM Results')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='lower right')
        plt.ylim([-160, 160])
        plt.xlim([-160, 160])
        plt.show()

        # # Plot batch accuracy
        # plt.plot(batch_accuracy, 'k-', label='Accuracy')
        # plt.title('Batch Accuracy')
        # plt.xlabel('Generation')
        # plt.ylabel('Accuracy')
        # plt.legend(loc='lower right')
        # plt.show()

        # # Plot loss over time
        # plt.plot(loss_vec, 'k-')
        # plt.title('Loss per Generation')
        # plt.xlabel('Generation')
        # plt.ylabel('Loss')
        # plt.show()

if __name__ == '__main__':
    main()