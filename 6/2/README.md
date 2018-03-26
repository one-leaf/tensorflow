参考 https://github.com/chineseocr/chinese-ocr

1. 编译 bbox, CPU版本需要修改 setup.py 和 __init__.py 关闭 cuda 模块
> cd lib/utils
> sh make.sh

2. 下载 VOCdevkit2007 到 data 目录 解压缩并改名为 VOCdevkit2007
> http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

3. 下载 VGG_imagenet.npy (528M) 复制到 data 目录下的 pretrain 目录
> https://drive.google.com/uc?id=0ByuDEGFYmWsbNVF5eExySUtMZmM&export=download
> http://bingxiang.oss-cn-shanghai.aliyuncs.com/VGG_imagenet.npy

4. 修改lib程序兼容 python3
> modify lib/fast_rcnn/config.py
> modify lib/datasets/pascal_voc.py

