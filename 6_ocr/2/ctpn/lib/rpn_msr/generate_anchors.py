import numpy as np

# 根据框的大小，按照标准16，分别计算所有框的坐标
def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors

# 计算中心点，然后根据高宽计算出box的坐标
def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5       # 15 * 0.5 = 7.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5       # 15 * 0.5 = 7.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin    # 7.5 - 16/2 = -0.5
    scaled_anchor[2] = x_ctr + w / 2  # xmax    # 7.5 + 16/2 = 15.5
    scaled_anchor[1] = y_ctr - h / 2  # ymin    # 7.5 - 11/2 = 2
    scaled_anchor[3] = y_ctr + h / 2  # ymax    # 7.5 + 11/2 = 13
    return scaled_anchor

# 根据一个像素，产生锚点框的列表，在ctpn中直接定义了等宽为16的10个不同高度的框的坐标
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes)

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
