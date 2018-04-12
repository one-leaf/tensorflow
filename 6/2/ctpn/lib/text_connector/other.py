import numpy as np


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2]=threshold(boxes[:, 0::2], 0, im_shape[1]-1)
    boxes[:, 1::2]=threshold(boxes[:, 1::2], 0, im_shape[0]-1)
    return boxes


# 文本框的表示图
# 001000
# 000100
# 000010
# 000000
# 000000
# 000000
class Graph:
    def __init__(self, graph):
        self.graph=graph

    def sub_graphs_connected(self):
        sub_graphs=[]
        # 按高度循环，取出所有
        for index in range(self.graph.shape[0]):
            # 如果当前列没有真值 并且当前行有真值
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v=index
                # 增加第一个
                sub_graphs.append([v])
                # 感觉这个代码有问题 ？？？
                # 本意是不断往下找有没有同一个文本框的，如果断了就表示没有了
                while self.graph[v, :].any():
                    # 返回当前行的一个为真的索引
                    v=np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


