import numpy as np
import cv2
from .config import cfg
from ..utils.blob import im_list_to_blob


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    # 将图片数据中值化
    im_orig -= cfg.PIXEL_MEANS

    # 取边长最大和最小值 im_shape (300,300,3)
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    # 缩放图片， cfg.TEST.SCALES：（600，）cfg.TEST.MAX_SIZE：1000，
    # 先按最小边缩放到 600，如果最大边大于1000，则按最大边缩放到1000来计算
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    # 这里只缩放一次，所以只有一张图片， blob shape (1,H,W,3)
    blob = im_list_to_blob(processed_ims)
    # 返回图片和缩放比例
    return blob, np.array(im_scale_factors)


def _get_blobs(im, rois):
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors

# 测试 ctpn
def test_ctpn(sess, net, im, boxes=None):
    # 返回缩放后的图片和缩放比，{‘data’:image, scale}
    # 钓鱼参数中，boxes 没有意义
    blobs, im_scales = _get_blobs(im, boxes)

    # 参数图片信息，{'data':image, scale, 'im_info':[H,W,scale]}
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    # forward pass
    # 由于是测试将 keep_prob 设置为1，并组装 tf 的 输入参数
    if cfg.TEST.HAS_RPN:
        feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

    rois = sess.run([net.get_output('rois')[0]],feed_dict=feed_dict)
    rois=rois[0]

    scores = rois[:, 0]
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]
    return scores,boxes
