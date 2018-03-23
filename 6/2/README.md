参考 https://github.com/chineseocr/chinese-ocr

1. 编译 bbox, CPU版本需要修改 setup.py 和 __init__.py 关闭 cuda 模块
> cd lib/utils
> sh make.sh

2. 下载 VOCdevkit2007 到 data 目录 解压缩并改名为 VOCdevkit2007
> http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

