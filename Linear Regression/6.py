import tensorflow as tf

# 求解 双十一预测 
def neural_networks():
    x = tf.placeholder(tf.float32, [None], name='x')
    y = tf.placeholder(tf.float32, [None], name='y')   
 
    ws = [tf.Variable(tf.random_normal([1])) for i in range(3)]
    b = tf.Variable(tf.random_normal([1]))
    c = tf.Variable(tf.random_normal([1]))
    y_pred = 0
    for i in range(3):
        y_pred += tf.multiply(ws[i], tf.pow(x+c,i+1))
    y_pred += b
 
    cost = tf.reduce_sum(tf.square(y_pred-y)) + tf.nn.l2_loss(ws)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    return x, y, y_pred, optimizer, cost
 
if __name__ == '__main__':
    X=(1,2,3,4,5,6,7,8,9,10)
    Y=(0.5,9.36,52,191,350,571,912,1207,1682,2135)
    x, y, y_pred, optimizer, cost = neural_networks()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500000):
            _, loss = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
            if i % 50000 == 0:
                print(i, loss)
        pred = sess.run(y_pred, feed_dict={x: [11]})
        print('2019双十一预测：', pred[0], '实际值：', 2684, '准确率：', 1-abs(2684-pred[0])/2684)

        X=(1,2,3,4,5,6,7,8,9,10,11)
        Y=(0.5,9.36,52,191,350,571,912,1207,1682,2135,2684)
        for i in range(50000):
            _, loss = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
        pred = sess.run(y_pred, feed_dict={x: [12]})
        print('2020双十一预测：', pred[0])                