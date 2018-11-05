import numpy as np
import matplotlib.pyplot as plt
import random

# 定义多少个质心
kCount = 6

# 按质心产生随机数据
def loadDataSet(count=200):
    center_point = np.random.random((kCount,2))
    data = np.zeros((count,2))
    for i in range(count):
        data[i,:]= center_point[i%kCount,:]+np.random.normal(0,0.05,(2))
    return data

# 从数据集中随机选取质心数据返回
def initCentroids(dataSet):
    dataSet = list(dataSet)
    return random.sample(dataSet, kCount)

# 计算向量之间的欧式距离
# 这里可以简化不需要再开方
def calcuDistance(vec1, vec2):
    return sum(np.square(vec1 - vec2)) 
    # return np.sqrt(sum(np.square(vec1 - vec2)))

# 计算item与质心的距离，找出距离最小的，并加入相应的簇类中
def minDistance(dataSet, centroidList):
    clusterDict = dict() 
    k = len(centroidList)
    for item in dataSet:
        vec1 = item
        minDist = np.inf
        minIndex = -1
        for i in range(k):
            vec2 = centroidList[i]
            distance = calcuDistance(vec1, vec2)  
            if distance < minDist:
                minDist = distance
                minIndex = i  
        if minIndex not in clusterDict.keys():
            clusterDict.setdefault(minIndex, [])
        clusterDict[minIndex].append(item)  
    return clusterDict  

# 将每个簇的中心点作为新质心
def getCentroids(clusterDict):
    centroidList = []
    for key in clusterDict.keys():
        centroid = np.mean(clusterDict[key], axis=0)
        centroidList.append(centroid)
    return centroidList  

# 展示聚类结果
def showCluster(centroidList, clusterDict):
    colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'oc', 'om'] 
    centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dc', 'dm']
    for key in clusterDict.keys():
        plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize=10) 
        for item in clusterDict[key]:
            plt.plot(item[0], item[1], colorMark[key])
    plt.show()

def main():
    dataSet = loadDataSet()
    centroidList = initCentroids(dataSet)
    clusterDict = minDistance(dataSet, centroidList)
    showCluster(centroidList, clusterDict)

    for i in range(200):
        centroidList = getCentroids(clusterDict)
        clusterDict = minDistance(dataSet, centroidList)
    
    showCluster(centroidList, clusterDict)

if __name__ == '__main__':
    main()


