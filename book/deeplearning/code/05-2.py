import numpy as np
import matplotlib.pyplot as plt
import random

# 定义多少个质心
k_count = 6
# 是否是k聚类++算法
is_k_means_plus = True

# 按质心产生随机数据
def loadDataSet(count=200):
    center_point = np.random.random((k_count,2))
    data = np.zeros((count,2))
    for i in range(count):
        data[i,:]= center_point[i%k_count,:]+np.random.normal(0,0.05,(2))
    return data

# 从数据集中选取初始化质心数据返回
def init_k_points(dataSet):
    dataSet = list(dataSet)
    if not is_k_means_plus:
        return random.sample(dataSet, k_count)
    else:
        k_points = []
        k_points.append(random.choice(dataSet))
        for i in range(1, k_count):
            d=[]
            for item in dataSet:
                vec1 = item
                d.append(np.inf)
                for point in k_points:
                    vec2 = point
                    distance = calc_distance(vec1, vec2)
                    if distance<d[-1]:
                        d[-1] = distance
            k_points.append(dataSet[d.index(max(d))])
        return k_points


# 计算向量之间的欧式距离
def calc_distance(vec1, vec2):
    return np.sqrt(sum(np.square(vec1 - vec2)))

# 计算item与质心的距离，找出距离最小的，并加入相应的簇类中
def minDistance(dataSet, k_points):
    cluster_dict = dict() 
    for item in dataSet:
        vec1 = item
        minDist = np.inf
        minIndex = -1
        for i in range(len(k_points)):
            vec2 = k_points[i]
            distance = calc_distance(vec1, vec2)  
            if distance < minDist:
                minDist = distance
                minIndex = i  
        if minIndex not in cluster_dict.keys():
            cluster_dict.setdefault(minIndex, [])
        cluster_dict[minIndex].append(item)  
    return cluster_dict  

# 将每个簇的中心点作为新质心
def getCentroids(cluster_dict):
    k_points = []
    for key in cluster_dict.keys():
        point = np.mean(cluster_dict[key], axis=0)
        k_points.append(point)
    return k_points  

# 展示聚类簇结果
def show_cluster(k_points, cluster_dict):
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'oc', 'om'] 
    k_point_mark = ['dr', 'db', 'dg', 'dk', 'dy', 'dc', 'dm']
    for key in cluster_dict.keys():
        plt.plot(k_points[key][0], k_points[key][1], k_point_mark[key], markersize=8, fillstyle="none") 
        for item in cluster_dict[key]:
            plt.plot(item[0], item[1], colorMark[key], markersize=3)
    plt.show()

def main():
    dataSet = loadDataSet()
    k_points = init_k_points(dataSet)
    cluster_dict = minDistance(dataSet, k_points)
    show_cluster(k_points, cluster_dict)

    for i in range(200):
        k_points = getCentroids(cluster_dict)
        cluster_dict = minDistance(dataSet, k_points)
    
    show_cluster(k_points, cluster_dict)

if __name__ == '__main__':
    main()


