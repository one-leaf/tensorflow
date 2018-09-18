# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from .network import Network
from ..fast_rcnn.config import cfg

# VGG 的训练模型
class VGGnet_train(Network):
    '''
    data : 输入图片
    im_info ：图片的高宽和缩放比例， image info
    gt_boxes : 图片中的框box，前4位是box的坐标，最后一位是box的类别， ground truth boxes 正确的标注数据
    gt_ishard ：是否为难以辨识的物体， 主要指要结体背景才能判断出类别的物体。虽有标注， 但一般忽略这类物体
    dontcare_areas ：don’t care areas，不关心的区域，也就是除开 前景，背景 之后的第三个分类，忽略掉 
    keep_prob ：训练时的 dropout 一般取值： 0.5
    trainable ： 是否可参与训练，也就是是否列入保存参数的范围
    '''
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes,\
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable
        self.setup()

    def setup(self):

        # n_classes = 2
        # 对于ctpn，分类只有 2 类，文字和非文字
        n_classes = cfg.NCLASSES
        # anchor_scales = [8, 16, 32]
        # 参数的box缩放类别，这里修改为 [16]， 默认一个点产生 9个框，这里只有3个框
        anchor_scales = cfg.ANCHOR_SCALES
        # 图片的采样率，图片经过4层 pool = 2 * 2 * 2 * 2 == 16 ，所以对应的 H,W 坐标，分别*16，就可以换算为 data 上对应的坐标
        _feat_stride = [16, ]

        # VGG16的网络结构
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 512, 1, 1, name='conv4_3')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3'))

        # 输出 shape， 这个是按原图说的 [N, H/16, W/16, 512]，不过对于 ctpn， N = 1， 也就是一次只学习一张图片
        # 后续按 [1, H, W, 512] 备注, 
        
        #========= RPN ============
        # 这里加入了一层3x3的窗口采样，这样每个点的值就包括了3*3的信息，如果太大或太小都造成计算复杂或信息不完整
        (self.feed('conv5_3')
             .conv(3,3,512,1,1,name='rpn_conv/3x3'))

        # 引入了 bilstm 模型
        # 按 [1 * H, W, C] ==> bilstm (128单元) ==> [1 * H * W, 2 * 128] ==> FC(512) ==> [1, H, W, 512]
        (self.feed('rpn_conv/3x3').Bilstm(512,128,512,name='lstm_o'))

        # bbox 位置偏移， 原来的faster rcnn是 9 * 4
        # [1, H, W, 512] ==> FC(40) ==> [1, H, W, 10 * 4] 每个坐标取10个框        
        (self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))

        # 分类， 原来的faster rcnn是 9 * 2
        # [1, H, W, 512] ==> FC(20) ==> [1, H, W, 10 * 2] 每个坐标取10个框        
        (self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 2,name='rpn_cls_score'))

        # shape is (1, H, W, 10*2) -> (1, H, W*10, 2)
        # 给之前得到的 score 进行 softmax ，得到 0-1 (bg / fg) 的得分
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name = 'rpn_cls_score_reshape')
             .spatial_softmax(name='rpn_cls_prob'))
             
        # generating training labels on the fly
        # 给每个 anchor 计算分类标签，并获得实际偏移量坐标
        # output: rpn_labels (1 x H x W x A, 2)                 分类
        #         rpn_bbox_targets (1 x H x W x A, 4)           回归
        #         rpn_bbox_inside_weights (1 x H x W x A, 4)    正负样本权重开关
        #         rpn_bbox_outside_weights (1 x H x W x A, 4)   正负样本权重
        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))


