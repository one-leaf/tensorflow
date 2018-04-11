import numpy as np

# 计算两个box的偏移量
def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    # 输入为 （n, 4）==> [n, [xmin, ymin, xmax, ymax]]
    # n 不等于 H * W， 删除了超过边框的数据，只包括了有效的的框
    # 框的宽度
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    # 框的高度
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    # 框的中心坐标
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    # 检查这一组中间最小的宽度和高度都必须大于 0.1
    assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        'Invalid boxes found: {} {}'. \
            format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    # 真实的框的宽度、高度和中心点坐标
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    # 计算中心点的偏移量比。
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    # 计算高宽的缩放比，取对数
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    # 返回 (n, 4)
    # vstack ==> (4, n), transpose ==> (n, 4)
    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets

# 由box和偏移量，求另外的一个box
def bbox_transform_inv(boxes, deltas):

    boxes = boxes.astype(deltas.dtype, copy=False)

    # boxes = [H*W*10, [xmin, ymin, xmax, ymax]]
    # 得到现有box的宽度、高度和中心坐标
    # widths , heights, ctr_x, ctr_y : [H*W*10]
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # deltas : [H*W*10 , 4]
    # dx,dy,dw,dh : [H*W*10, 1]
    # ::4 按步长为 4 取值
    # dx，dy 中心偏移比
    # dw, dh 高宽缩放比取对数 
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # pred_ctr_x : [H*W*10] => [H*W*10,1]
    # ctpn 中 x不变的，宽度固定为16
    pred_ctr_x = ctr_x[:, np.newaxis]

    # [H*W*10,1] * [H*W*10,1]+[H*W*10,1]
    # 预测的偏移比 * box的高度+中心位置，就是预测的中心坐标 y
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    # ctpn 中 w不变固定16
    pred_w = widths[:, np.newaxis]
    # 高度需要取指数，再乘以框的高度，获得实际高度
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    # 返回 [H*W*10 , 4] 所有框的预测坐标
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

# 修正框的边界不能超过图片的边界
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
