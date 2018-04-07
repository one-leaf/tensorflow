参考 https://github.com/chineseocr/chinese-ocr

1. 编译 bbox, CPU版本需要修改 setup.py 和 __init__.py 关闭 cuda 模块
> cd lib/utils
> sh make.sh

2. 下载 VOCdevkit.zip 到 data 目录 解压缩并改名为 VOCdevkit2007
> https://pan.baidu.com/s/1kUNTl1l

3. 下载 VGG_imagenet.npy (528M) 复制到 data 目录下的 pretrain 目录
> https://pan.baidu.com/s/1kUNTl1l
> https://drive.google.com/uc?id=0ByuDEGFYmWsbNVF5eExySUtMZmM&export=download
> http://bingxiang.oss-cn-shanghai.aliyuncs.com/VGG_imagenet.npy

4. 修改lib程序兼容 python3
> 修改 lib/fast_rcnn/config.py
> 修改 lib/datasets/pascal_voc.py

5. 修改CPU版本或者用GPU版本

6. 编译 lib/utils
> 修改 mask.sh 改为 python3
> cd lib/utils
> ./make.sh

7. 运行train.net
> sh ctpn/train_net.py

文档：

http://rowl1ng.com/tech/faster-rcnn.html