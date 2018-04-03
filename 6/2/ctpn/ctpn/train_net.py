import pprint
import sys
import os.path

sys.path.append(os.getcwd())
this_dir = os.path.dirname(__file__)

from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg

if __name__ == '__main__':
    cfg_from_file('ctpn/text.yml')
    print('Using config:')
    pprint.pprint(cfg)

    # 获得voc2007的数据集
    # {
    #     year:’2007’
    #     image _set:’trainval’
    #     devkit _path:’data/VOCdevkit2007’
    #     data _path:’data /VOCdevkit2007/VOC2007’
    #     classes:(…)_如果想要训练自己的数据，需要修改这里_
    #     class _to _ind:{…} _一个将类名转换成下标的字典 _  建立索引0,1,2....
    #     image _ext:’.jpg’
    #     image _index: [‘000001’,’000003’,……]_根据trainval.txt获取到的image索引_
    #     roidb _handler: <Method gt_roidb >
    #     salt:  <Object uuid >
    #     comp _id:’comp4’
    #     config:{…}
    # }
    imdb = get_imdb('voc_2007_trainval')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))

    # roidb 是 imdb 的一个 类型为 dict 的属性
    # 元素如下：
    # boxes         一个二维数组，每一行存储 xmin ymin xmax ymax
    # gt_overlaps   是一个二维数组，共有 num_classes(即类的个数)行，每一行对应的box的类索引处值为1，其余皆为0，后来被转成了稀疏矩阵
    # gt_classes    存储了每个box所对应的类索引(类数组在初始化函数中声明)
    # flipped       为false 代表该图片还未被翻转(后来在train.py里会将翻转的图片加进去，用该变量用于区分
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    log_dir = get_log_dir(imdb)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))

    device_name = '/gpu:0'
    print(device_name)

    # 采用VGG网络
    network = get_network('VGGnet_train')

    # 开始训练
    train_net(network, imdb, roidb,
              output_dir=output_dir,
              log_dir=log_dir,
              pretrained_model='data/pretrain/VGG_imagenet.npy',
              max_iters=int(cfg.TRAIN.max_steps),
              restore=bool(int(cfg.TRAIN.restore)))
