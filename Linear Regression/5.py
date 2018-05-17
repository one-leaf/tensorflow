import tensorflow as tf
# 求解 () () () 2 4 6 7 8 
def neural_networks():
    x = tf.placeholder(tf.float32, [None], name='x')
    y = tf.placeholder(tf.float32, [None], name='y')   

    ws = [tf.Variable(tf.random_normal([1])) for i in range(5)]
    b = tf.Variable(tf.zeros(1))
    c = tf.Variable(tf.zeros(1))
    y_pred = 0
    for i in range(5):
        y_pred += tf.multiply(ws[i], tf.pow(x+c,i))
    y_pred += b

    cost = tf.reduce_sum(tf.square(y_pred-y))  
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
    return x, y, y_pred, optimizer, cost

if __name__ == '__main__':
    X=(4,5,6,7,8)
    Y=(2,4,6,7,8)
    x, y, y_pred, optimizer, cost = neural_networks()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000000):
            _, loss = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
            if i % 1000 == 0:
                pred = sess.run(y_pred, feed_dict={x: [1,2,3]})
                print(i, loss, pred)
           
