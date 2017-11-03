这里是逻辑回归的一些测试

主要数据采用 MNIST ，数据来源 ： http://yann.lecun.com/exdb/mnist/

如果数据无法下载，可以手工下载到 data 目录：

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz



S(t) = 1 / (1 + exp(-t))   ==> [0,1]