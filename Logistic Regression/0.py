import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

curr_dir = os.path.dirname(__file__)
mnist = input_data.read_data_sets(os.path.join(curr_dir,"data"), one_hot=True)

# 55000 组图片和标签, 用于训练
def getBatch(batchSize):
    batch_x, batch_y = mnist.train.next_batch(batchSize)
    return batch_x, batch_y  

# 5000 组图片和标签, 用于迭代验证训练的准确性
def getValidationImages():
    return mnist.validation.images, mnist.validation.labels	

# 10000 组图片和标签, 用于最终测试训练的准确性
def getTestImages():
    return mnist.test.images, mnist.test.labels

# 10*10 显示 100 张图片
def plot_10_by_10_images(images):
    fig = plt.figure()
    images = [image[3:25, 3:25] for image in images]
    for x in range(0,10):
        for y in range(0,10):
            ax = fig.add_subplot(10, 10, 10*y+x+1)
            ax.matshow(images[10*y+x], cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

if __name__ == '__main__':
    batch_x, batch_y = getBatch(100)
    images =  [np.reshape(f, (28, 28)) for f in batch_x]
    plot_10_by_10_images(images)

