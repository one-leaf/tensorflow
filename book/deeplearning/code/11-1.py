# F 和 PR

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def getData(size=100):
    x = np.random.random((size, 2))*2-1
    y = np.power(x, 2)
    y = np.sum(y, axis=1)
    y[y > 0.5] = 1
    y[y < 1]   = 0
    y = np.eye(2)[y.astype(np.int32)]
    return x+0.05*np.random.normal(size=(size,2)), y

class network():
    def __init__(self, lstm=True):
        self.x = tf.placeholder(tf.float32, [None, 2], name='x')
        self.y = tf.placeholder(tf.float32, [None, 2], name='y')
        layer = tf.layers.dense(self.x, 100, activation=tf.nn.relu)
        layer = tf.layers.dense(layer, 100, activation=tf.nn.relu)
        self.pred = tf.layers.dense(layer, 2) 
        self.loss = tf.losses.softmax_cross_entropy(self.y, self.pred)
        self.optimizer= tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1)), tf.float32))

def calc(y, pred):
    tp = 0          #true positives 被检索到的正类
    fp = 0          #false positives 被检索到的负类
    fn = 0          #false negatives 未被检索到的正类
    tn = 0          #true negatives 未被检索到的负类
    for i in range(len(y)):
        if y[i]==1:
            if pred[i]==1:
                tp+=1
            else:
                fn+=1
        else:
            if pred[i]==1:
                fp+=1
            else:
                tn+=1
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    f=2*p*r/(p+r)
    return tp,fp,fn,tn,p,r,f

def clac_roc(y_true, pos_prob):
    pos = y_true[y_true==1]
    neg = y_true[y_true==0]
    threshold = np.sort(pos_prob)[::-1]        # 按概率大小逆序排列
    y = y_true[pos_prob.argsort()[::-1]]
    tpr_all = [0] ; fpr_all = [0]
    tpr = 0 ; fpr = 0
    x_step = 1/float(len(neg))
    y_step = 1/float(len(pos))
    y_sum = 0                                  # 用于计算AUC
    for i in range(len(threshold)):
        if y[i] == 1:
            tpr += y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr += x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum += tpr
    return tpr_all,fpr_all,y_sum*x_step 

def calc_pr(y_true, pos_prob):
    pos = y_true[y_true==1]
    threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    recall = [] ; precision = []
    tp = 0 ; fp = 0
    auc = 0
    for i in range(len(threshold)):
        if y[i] == 1:
            tp += 1
            recall.append(tp/len(pos))
            precision.append(tp/(tp+fp))
            auc += (recall[i]-recall[i-1])*precision[i]
        else:
            fp += 1
            recall.append(tp/len(pos))
            precision.append(tp/(tp+fp))
    return precision,recall,auc

def main():
    net = network(True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_size = 100
        p_list=[]
        r_list=[]
        f_list=[]
        acc_list=[]
        fpr_list=[]
        for epoch in range(200):
            batch_xs, batch_ys = getData(batch_size)
            loss, acc, pred, _= sess.run([net.loss,net.acc,net.pred, net.optimizer], feed_dict={net.x: batch_xs, net.y: batch_ys})
            print(epoch, "loss:", loss,"acc:", acc)

            x_data, y_data = getData(10000)
            pred = sess.run(net.pred, feed_dict={net.x: x_data}) 
            pred = np.argmax(pred,1)  
            y = np.argmax(y_data, 1)
            tp,fp,fn,tn,p,r,f = calc(y,pred)
            acc_list.append(np.mean(np.equal(y,pred)))
            p_list.append(p)
            r_list.append(r)
            f_list.append(f)
            fpr_list.append(fp/(fp+tn))

        plt.figure()
        x_data, y_data = getData(10000)
        y = np.argmax(y_data, 1)
        y_pred = sess.run(net.pred, feed_dict={net.x: x_data})
        pred = np.argmax(y_pred, 1)
        pred[pred!=y]=2
        plt.scatter(x_data[:, 0], x_data[:, 1], marker='o', c=pred, s=3)

        plt.figure()
        x = np.linspace(1, 200, 200)
        plt.plot(x, acc_list, label='Accuracy Rate')
        plt.plot(x, p_list, label='True Precision Rate')
        plt.plot(x, r_list, label='Recall Rate')
        plt.plot(x, f_list, label='F-Measure')
        plt.plot(x, fpr_list, label='False Positive Rate')
        plt.legend()

        pred = y_pred[:,1]
        plt.figure()
        p,r,auc =calc_pr(y, pred)
        plt.plot(r, p, label='P-R (auc: %s)'%auc)
        plt.xlabel('Recall Rate')
        plt.ylabel('Precision Rate')
        plt.legend()

        plt.figure()
        tpr,fpr,auc =clac_roc(y, pred)
        plt.plot(fpr, tpr, label='ROC (auc: %s)'%auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()