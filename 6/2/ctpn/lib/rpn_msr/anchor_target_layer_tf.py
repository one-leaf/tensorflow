# -*- coding:utf-8 -*-
import numpy as np
import numpy.random as npr
from .generate_anchors import generate_anchors
from ..utils.bbox import bbox_overlaps, bbox_intersections
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

DEBUG = False
def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride = [16,], anchor_scales = [16,]):
    """
    将锚点分配给真实的目标，产生锚点的分类标签和边界回归目标
    输入参数：
    ----------
    rpn_cls_score: (1, H, W, Ax2) bg/fg 之前 conv 层定义的分类
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class] 输入的一堆box框和分类
    gt_ishard: (G, 1), 1 or 0 是否很难识别，1是困难，0是容易
    dontcare_areas: (D, 4), 不需要理会的 box，D 可能为 0
    im_info: [image_height, image_width, scale_ratios] 列表，这里只有1张图片
    _feat_stride: VGG缩放后的特征图和输入的原始图的比例，VGG网络为16
    anchor_scales: 锚点框的缩放大小，ctpn 只有一个 [16,]
    ----------
    返回值：
    ----------
    rpn_labels : (HxWxA, 1), 对于每个锚点框, 0 为 bg 背景, 1 fg 前景, -1 dontcare 抛弃
    rpn_bbox_targets: (HxWxA, 4), 锚点框的坐标 和 gt_boxes 的偏移量, 这个就是回归目标
    rpn_bbox_inside_weights: (HxWxA, 4) 每个框的权重，在 cfg 中定义
    rpn_bbox_outside_weights: (HxWxA, 4) 用于平衡 fg 和 bg 的正反样本的个数， 方便训练
    """
    # 在ctpn中直接定义了等宽为16的10个不同高度的锚点框 shape : [10, 4]
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0] # 10个anchor

    if DEBUG:
        print('anchors:')
        print(_anchors)
        print('anchor shapes:')
        print(np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        )))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # 是否允许 box 压住目标边缘的距离， 对于文字检测ctpn，这里为0，即框等于实际文字的范围
    _allowed_border =  0

    im_info = im_info[0] # 取第一张原始图像的高宽及通道数，也只有一张图片

    # 在feature-map上定位anchor，并加上delta，得到在实际图像中anchor的真实坐标
    # 算法:
    # for each (H, W) 每个位置坐标 i
    #   以 i 为中心 产生 10 个 锚窗
    #   按10个锚窗预测 bbox 的 偏移量
    # 过滤掉超出图片的锚窗
    # 检测重叠样本

    # 只支持每次训练1张图片
    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # 读取 VGG之后的特征图的高宽
    # map of shape (1, H, W，C)
    height, width = rpn_cls_score.shape[1:3]#feature-map的高宽

    if DEBUG:
        print('AnchorTargetLayer: height', height, 'width', width)
        print('')
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))
        print('height, width: ({}, {})'.format(height, width))
        print('rpn: gt_boxes.shape', gt_boxes.shape)
        print('rpn: gt_boxes', gt_boxes)

    # 1. Generate proposals from bbox deltas and shifted anchors
    # 产生锚点的原始对应坐标
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    # shift_x, shift_y shape = [ h, w ]
    # shift_x = [[0,16,32,...],[0,16,32,...],...]
    # shift_y = [[0,0,0,...],[16,16,16,...],...]
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # in W H order

    # 获得原图的采样点重复2次，每个采样点相隔16
    # shifts = array([[  0,  16,  32, ..., 464, 480, 496],
    #       [  0,   0,   0, ..., 272, 272, 272],
    #       [  0,  16,  32, ..., 464, 480, 496],
    #       [  0,   0,   0, ..., 272, 272, 272]]).transpose()
    # shifts.shape=[H x W, 4] 
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()#生成feature-map和真实image上anchor之间的偏移量

    # K is H x W 
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # A = 10个anchor
    A = _num_anchors
    # K = feature-map的宽乘高的大小
    K = shifts.shape[0]
    # 得到所有的锚框 (1, 10, 4) + (H*W, 1, 4) => (H*W, 10, 4)
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))#相当于复制宽高的维度，然后相加
    # 所有的锚框（H*W, 10, 4）=> (H*W*10, 4)
    all_anchors = all_anchors.reshape((K * A, 4))
    # 全部锚框的个数为 H*W*10
    total_anchors = int(K * A)

    # only keep anchors inside the image
    # 仅保留那些还在图像内部的锚框，超出图像的都删掉
    # inds_inside shape 符合条件的(K * A) => (I)
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]

    if DEBUG:
        print('total_anchors', total_anchors)
        print('inds_inside', len(inds_inside))

    # keep only inside anchors
    # 保留那些在图像内的anchor
    # anchors shape (I, 4)
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print('anchors.shape', anchors.shape)

    #至此，anchor准备好了
    #--------------------------------------------------------------
    # label: 1 是 正样本, 0 是 负样本, -1 是 忽略
    # (I)
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1) #初始化label，均为-1

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt), shape is I x G
    # 计算anchor和gt-box的overlap，用来给anchor上标签
    # 假设anchors有I个，gt_boxes有G个，返回的是一个（I,G）的数组
    # 存放每一个anchor和每一个gtbox之间的overlap
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))

    # 找到每一个锚框的 overlap 最大值
    # shape: (I)
    # 这个是最大值的 index
    argmax_overlaps = overlaps.argmax(axis=1)
    # 这个 shape 也是 (I), 是最大值的 value
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

    # 找到每一个 gt_boxes 的 overlap 最大值
    # shape: (G)
    # 这个是最大值的 index    
    gt_argmax_overlaps = overlaps.argmax(axis=0) 
    # 这个 shape 也是 (G), 是最大值的 value
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    # 这个貌似没有意义，值和前面的 gt_argmax_overlaps 一样，都是在 overlaps 0维度上的索引
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    # 如果一个锚框同时满足正样本和负样本的条件设置为负样本，这个默认关闭
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # 将 overlaps 值小于0.3的设置为负样本，也就是背景 0 
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # 最大的overlap设置为 前景
    labels[gt_argmax_overlaps] = 1 
    # 大于 0.7 的也设置为 前景
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    # 如果一个锚框同时满足正样本和负样本的条件设置为负样本，这个默认关闭
    # 这个代码有问题将导致 cfg.TRAIN.RPN_CLOBBER_POSITIVES 失效，无论如何都会将小于 0.3 的都设置为负样本
    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # 将 overlaps 值小于0.3的设置为负样本，也就是背景 0 
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # 排除不关心的区域
    # 在ctpn模型中，暂时不考虑有 doncare_area 的存在
    if dontcare_areas is not None and dontcare_areas.shape[0] > 0:
        # intersec shape is D x A
        intersecs = bbox_intersections(
            np.ascontiguousarray(dontcare_areas, dtype=np.float), # D x 4
            np.ascontiguousarray(anchors, dtype=np.float) # A x 4
        )
        intersecs_ = intersecs.sum(axis=0) # A x 1
        # 将计算出来的和不关心区域的值大于 >0.5 都忽略掉
        labels[intersecs_ > cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI] = -1

    # 在ctpn中 也暂时不考虑难样本的问题，不需要排除困难样本
    # preclude hard samples that are highly occlusioned, truncated or difficult to see
    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
        assert gt_ishard.shape[0] == gt_boxes.shape[0]
        gt_ishard = gt_ishard.astype(int)
        gt_hardboxes = gt_boxes[gt_ishard == 1, :]
        if gt_hardboxes.shape[0] > 0:
            # H x A
            hard_overlaps = bbox_overlaps(
                np.ascontiguousarray(gt_hardboxes, dtype=np.float), # H x 4
                np.ascontiguousarray(anchors, dtype=np.float)) # A x 4
            hard_max_overlaps = hard_overlaps.max(axis=0)  # (A)
            labels[hard_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1
            max_intersec_label_inds = hard_overlaps.argmax(axis=1) # H x 1
            labels[max_intersec_label_inds] = -1 #

    # subsample positive labels if we have too many
    # 对正样本进行采样，如果正样本的数量太多的话
    # 限制正样本的数量不超过128个
    # TODO 这个后期可能还需要修改，毕竟如果使用的是字符的片段，那个正样本的数量是很多的。
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)#随机去除掉一些正样本
        labels[disable_inds] = -1#变为-1

    # subsample negative labels if we have too many
    # 对负样本进行采样，如果负样本的数量太多的话
    # 正负样本总数是256，限制正样本数目最多128，
    # 如果正样本数量小于128，差的那些就用负样本补上，凑齐256个样本
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))

    # 至此， 上好标签，开始计算rpn-box的真值
    #--------------------------------------------------------------
    # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差）
    # shape = (I, 4) 这行代码没有用
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # anchors shape (I, 4) gt_boxes shape [I, 5], 最大值
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])


    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)#内部权重，前景就给1，其他是0

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:#暂时使用uniform 权重，也就是正样本是1，负样本是0
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights#外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print('means:')
        print(means)
        print('stdevs:')
        print(stds)

    # map up to original set of anchors
    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)#这些anchor的label是-1，也即dontcare
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)#这些anchor的真值是0，也即没有值
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)#内部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)#外部权重以0填充

    if DEBUG:
        print('rpn: max max_overlap', np.max(max_overlaps))
        print('rpn: num_positive', np.sum(labels == 1))
        print('rpn: num_negative', np.sum(labels == 0))
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print('rpn: num_positive avg', _fg_sum / _count)
        print('rpn: num_negative avg', _bg_sum / _count)

    # labels
    labels = labels.reshape((1, height, width, A))#reshap一下label
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))#reshape

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
