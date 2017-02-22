# coding=utf-8
# 卷积多层多项验证码识别

from generate_captcha import gen_captcha_text_and_image as captcha
import numpy as np
from utils import img2gray,img2vec,text2vec,vec2text
import tensorflow as tf

image_h=80
image_w=200
image_size=image_h*image_w
char_set="0123456789"
char_size=len(char_set)
captcha_size = 4

# 批量验证码数据
def get_batch(batch_size=128):
    batch_x = np.zeros([batch_size, image_size])
    batch_y = np.zeros([batch_size, char_size*captcha_size])
    for i in range(batch_size):
        text, image = captcha(char_set=char_set,captcha_size=captcha_size,width=image_w, height=image_h)
        batch_x[i,:] = img2vec(img2gray(image))
        batch_y[i,:] = text2vec(char_set,text)
    return batch_x, batch_y

# 采用随机数初始化权重函数
def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 采用随机数初始化偏置函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积(Convolution) 滑动步长为1的窗口，使用0进行填充
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化(Pooling) 在2*2的窗口内采用最大池化技术(max-pooling) 图像尺寸将缩小1半
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第一层卷积层  计算出32个特征映射(feature map)，对每个3*3的patch 前两维是patch的大小，第三维时输入通道的数目，最后一维是输出通道的数目。
W_conv1 = weight_varible([3, 3, 1, 32])
# 对每个输出通道加上了偏置(bias)
b_conv1 = bias_variable([32])

# 为了使得图片与计算层匹配，我们首先reshape输入图像x为4维的tensor，
# 第一维 -1 是不限个和 None 类似， 第2、3维对应图片的宽和高，最后一维对应颜色通道的数目，这里是黑白，所以为 1 ，如果图片为 RGB 则为3 。
x = tf.placeholder(tf.float32, [None, image_size])
x_image = tf.reshape(x, [-1, image_w, image_h, 1])

# 使用weight tensor对x_image进行卷积计算，加上bias，再应用到一个ReLU激活函数，最终采用最大池化。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积层 为了使得网络有足够深度，我们重复堆积一些相同类型的层。第二层将会有64个特征，对应每个3*3的patch。
W_conv2 = weight_varible([3, 3, 32, 64])
b_conv2 = bias_variable([64])

# 使用第一层的池化输出作为输入，最终再池化输出
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层 经过两次池化，输出尺寸为原始图片尺寸/16 ，加入一个神经元数目为1024的全连接层来处理所有的图像。
W_fc1 = weight_varible([image_h * image_w * 64 // 16, 1024])
b_fc1 = bias_variable([1024])
# 将最后的pooling层的输出reshape为一个一维向量，与权值相乘，加上偏置，再通过一个ReLu函数。
h_pool2_flat = tf.reshape(h_pool2, [-1, image_h * image_w * 64 // 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2_1 = weight_varible([1024, 1024])
b_fc2_1 = bias_variable([1024])
h_pool2_1_flat = tf.reshape(h_fc1, [-1, 1024])
h_fc2_1 = tf.nn.relu(tf.matmul(h_pool2_1_flat, W_fc2_1) + b_fc2_1)

W_fc2_2 = weight_varible([1024, 1024])
b_fc2_2 = bias_variable([1024])
h_pool2_2_flat = tf.reshape(h_fc1, [-1, 1024])
h_fc2_2 = tf.nn.relu(tf.matmul(h_pool2_2_flat, W_fc2_2) + b_fc2_2)

W_fc2_3 = weight_varible([1024, 1024])
b_fc2_3 = bias_variable([1024])
h_pool2_3_flat = tf.reshape(h_fc1, [-1, 1024])
h_fc2_3 = tf.nn.relu(tf.matmul(h_pool2_3_flat, W_fc2_3) + b_fc2_3)

W_fc2_4 = weight_varible([1024, 1024])
b_fc2_4 = bias_variable([1024])
h_pool2_4_flat = tf.reshape(h_fc1, [-1, 1024])
h_fc2_4 = tf.nn.relu(tf.matmul(h_pool2_4_flat, W_fc2_4) + b_fc2_4)

fc2 = tf.concat(1, [h_fc2_1,h_fc2_2,h_fc2_3,h_fc2_4])


# 为了减少过拟合程度，在输出层之前应用dropout技术（即随机丢弃某些神经元的输出结果）
keep_prob = tf.placeholder(tf.float32)
#h_fc_last_drop = tf.nn.dropout(h_fc4, keep_prob)

# 最终，我们用一个softmax层，得到类别上的概率分布。
W_fc_last = weight_varible([1024*4, char_size*captcha_size])
b_fc_last = bias_variable([char_size*captcha_size])
y_conv = tf.nn.softmax(tf.matmul(fc2, W_fc_last) + b_fc_last)
y_ = tf.placeholder(tf.float32, [None, char_size*captcha_size])

# 训练函数，采用了 AdamOptimizer 代替 之前的 GradientDescentOptimizer 
# 该函数需要增了额外的参数keep_prob在feed_dict中，以控制dropout的几率；
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
# 多输入和多分类问题，必须采用 sigmoid_cross_entropy_with_logits 函数
# 参考： http://weibo.com/ttarticle/p/show?id=2309404047468714166594
#cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y_))
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_conv, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 比较计算结果是否正确
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
# 统计准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = get_batch(50)
    if i % 10 == 0:
        train_accuacy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuacy))
    train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

# 最后测试准确率
batch_x_test, batch_y_test = get_batch(100)
print("test accuracy %g"%(accuracy.eval(feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.0})))
    