from .text_connect_cfg import Config as TextLineCfg
from .other import Graph
import numpy as np


class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """
    # 正序查同一水平的框
    def get_successions(self, index):
            box=self.text_proposals[index]
            results=[]
            # TextLineCfg.MAX_HORIZONTAL_GAP 50 最大长度
            # 从 xmin+1 的坐标到 min(xmin+50+1, W) 循环
            # 找到此宽度下的所有box
            # 如果在这范围内的其它框在同一个水平，则加入
            for left in range(int(box[0])+1, min(int(box[0])+TextLineCfg.MAX_HORIZONTAL_GAP+1, self.im_size[1])):
                adj_box_indices=self.boxes_table[left]
                for adj_box_index in adj_box_indices:
                    if self.meet_v_iou(adj_box_index, index):
                        results.append(adj_box_index)
                if len(results)!=0:
                    return results
            return results

    # 倒序查同一水平的框
    def get_precursors(self, index):
        box=self.text_proposals[index]
        results=[]
        # 从 xmin -1 到 max(xmin-50-1,0) 循环
        # 返回同一水平的框
        for left in range(int(box[0])-1, max(int(box[0]-TextLineCfg.MAX_HORIZONTAL_GAP), 0)-1, -1):
            adj_box_indices=self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results

    # 如果倒序查的结果也是这个框的概率最大，认为的确是有效的
    def is_succession_node(self, index, succession_index):
        precursors=self.get_precursors(succession_index)
        if self.scores[index]>=np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        # 得到两个box的高度
        # 计算两个框的最大的ymin 为 y0 和最小的 ymax 为 有y1
        # 返回重叠部分占最小高度的比 
        def overlaps_v(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            y0=max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1=min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1-y0+1)/min(h1, h2)

        # 计算两个框的高度比
        def size_similarity(index1, index2):
            h1=self.heights[index1]
            h2=self.heights[index2]
            return min(h1, h2)/max(h1, h2)

        # 如果高度重合 TextLineCfg.MIN_V_OVERLAPS 0.7 并且 高度比例 也大于 TextLineCfg.MIN_SIZE_SIM 0.7,
        # 则认为2个框是同一水平
        return overlaps_v(index1, index2)>=TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2)>=TextLineCfg.MIN_SIZE_SIM

    # 调用入口
    # text_proposals [k, 4]
    # scores [k, 1]
    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals=text_proposals
        self.scores=scores
        self.im_size=im_size
        self.heights=text_proposals[:, 3]-text_proposals[:, 1]+1

        # 按宽度创建了 [[],[],[],...] boxes_table
        # 将全部的框的序号 按框的x坐标放入 boxes_table
        # boxes_table [[1,2],[],[0,3,4],...]
        boxes_table=[[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table=boxes_table

        # 创建 graph (K, K) ,K 为 H*W的子集
        graph=np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)
        # 循环所有的框
        for index, box in enumerate(text_proposals):
            # 得到当前框同一水平的其它框
            # 假设真实的一个文本框包括的序号为 [0,1,2,3,4], 检查的像素跨度为3
            # 则: index = 0 successions 为 [1,2,3]，表示1，2，3 这4个框都是和0在同一个水平线，这里并不完整，最大只检查50个像素
            #     index = 1 successions 为 [2,3,4]， 
            #     index = 2 successions 为 [3,4]， 
            #     index = 3 successions 为 [4]， 
            successions=self.get_successions(index)
            if len(successions)==0:
                continue
            # 得到这一组框中最大概率的框的序号
            # 例如，假设：succession_index 为 2
            succession_index=successions[np.argmax(scores[successions])]

            # 如果倒序检查也是这个框的概率最大，则认为这个是对的
            # 标记当前框和最大概率是True
            # 得到类似下面这样的正方形图
            # 注意，但index =2 时，如果3的概率和2一样，这样也会包括到graph中，这里假定2的概率比2小，所以没有，但这种算法不会丢失信息吗？？？
            # 001000
            # 001000
            # 000000
            # 000000
            # 000000
            # 000000
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index]=True
        return Graph(graph)
