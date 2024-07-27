import random
import sys

import numpy as np

"""
实现基于kmeans的类内距离计算，筛选优质类别
"""

class KMeansCluster:     # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(self.ndarray, self.cluster_num)
        self.count = 0

    def cluster(self):
        self.count += 1
        print("第{}次聚类".format(self.count))
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    """
    初始化逻辑
    1. 初始时选取2K个点作为质心
    2. 计算这2K个点聚类每个类内平均距离
    3. 根据平均距离从大到小排序，舍弃前K个点，保留后面K个点作为质心
    """
    def __pick_start_point(self, ndarray, cluster_num):
        start_cluster_number = cluster_num * 2
        if start_cluster_number < 0 or start_cluster_number > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), start_cluster_number)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        # 计算每一类的类内平均距离
        tmp_cluster = []
        tmp_cluster_distance = []
        for i in range(start_cluster_number):
            tmp_cluster.append([])
            tmp_cluster_distance.append(0)
        for item in self.ndarray:
            index = -1
            for i in range(len(points)):
                # 计算每一个元素和中心点之间的距离加总
                distance = self.__distance(item, points[i])
                tmp_cluster_distance[i] += distance
            tmp_cluster[index] = tmp_cluster[index] + [item.tolist()]
        print("tmp_cluster_distance is:{}".format(tmp_cluster_distance))
        indexed_arr = [(value, index) for index, value in enumerate(tmp_cluster_distance)]
        print("排序前的indexed_arr为:{}".format(indexed_arr))
        # 根据元素值对列表进行排序
        indexed_arr.sort(key=lambda x: x[0])
        print("排序后的indexed_arr为:{}".format(indexed_arr))
        # 取出前cluster_num个元素的索引
        original_indexes = [index for value, index in indexed_arr[:cluster_num]]
        print("取前cluster_num个元素的索引列表为:{}".format(original_indexes))
        final_points = []
        for index in original_indexes:
            final_points.append(ndarray[index].tolist())
        return np.array(final_points)



x = np.random.rand(100, 8)
# print("x is {} \nand x shape is {}".format(x, x.shape))
kmeans = KMeansCluster(x, 10)
result, centers, distances = kmeans.cluster()
print("result is {}\nand result len is {}".format(result, len(result)))
print("centers is {}\nand centers shape is {}".format(centers, centers.shape))
print("distances is {}".format(distances))