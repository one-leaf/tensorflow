import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

curr_dir = os.path.dirname(__file__)
mnist = input_data.read_data_sets(os.path.join(curr_dir,"data"), one_hot=True)

def getBatch(batchSize):
    batch_x, batch_y = mnist.train.next_batch(batchSize)
    return batch_x, batch_y  

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
    print(batch_x.shape)
    images =  [np.reshape(f, (-1, 28)) for f in batch_x]
    plot_10_by_10_images(images)


