#coding:utf-8
import numpy as np
from lib.fast_rcnn.nms_wrapper import nms
from lib.fast_rcnn.config import cfg
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented
from .text_connect_cfg import Config as TextLineCfg

# 文本框拼接
# 将一组不定高度，宽度为16的文本框拼接成连续的大框
class TextDetector:
    def __init__(self):
        self.mode= cfg.TEST.DETECT_MODE
        if self.mode == "H":
            self.text_proposal_connector=TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector=TextProposalConnectorOriented()


    # text_proposals 锚点框的坐标
    # scores 锚点框的概率
    # size 图片缩放后的size    
    def detect(self, text_proposals,scores,size):
        # 删除得分较低的proposal 将低于概率0.7的框都不要了
        keep_inds=np.where(scores>TextLineCfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

        # 按得分排序
        sorted_indices=np.argsort(scores.ravel())[::-1]
        text_proposals, scores=text_proposals[sorted_indices], scores[sorted_indices]

        # 对proposal做nms ，TEXT_PROPOSALS_NMS_THRESH : 0.2
        keep_inds=nms(np.hstack((text_proposals, scores)), TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

        # 获取检测结果
        text_recs=self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        # 最后检查框的高宽比，以及概率和最小宽度
        keep_inds=self.filter_boxes(text_recs)
        return text_recs[keep_inds]

    def filter_boxes(self, boxes):
        heights=np.zeros((len(boxes), 1), np.float)
        widths=np.zeros((len(boxes), 1), np.float)
        scores=np.zeros((len(boxes), 1), np.float)
        index=0
        for box in boxes:
            heights[index]=(abs(box[5]-box[1])+abs(box[7]-box[3]))/2.0+1
            widths[index]=(abs(box[2]-box[0])+abs(box[6]-box[4]))/2.0+1
            scores[index] = box[8]
            index += 1

        return np.where((widths/heights>TextLineCfg.MIN_RATIO) & (scores>TextLineCfg.LINE_MIN_SCORE) &
                          (widths>(TextLineCfg.TEXT_PROPOSALS_WIDTH*TextLineCfg.MIN_NUM_PROPOSALS)))[0]