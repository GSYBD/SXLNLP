import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

X = ...

k = 3
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


def calculate_intra_cluster_distance(X, labels, centroids):
    intra_distances = []
    for i in range(k):
        # 筛选出第i类的所有数据点
        cluster_points = X[labels == i]
        # 计算这些点到质心的距离
        distances = euclidean_distances(cluster_points, [centroids[i]])
        # 计算平均距离
        average_distance = np.mean(distances)
        intra_distances.append(average_distance)
    return intra_distances


intra_distances = calculate_intra_cluster_distance(X, labels, centroids)

best_cluster_index = np.argmin(intra_distances)
print(f"最优类别索引: {best_cluster_index}, 类内距离: {intra_distances[best_cluster_index]}")

threshold = 0.5
good_clusters = [i for i, d in enumerate(intra_distances) if d <= threshold]
print(f"基于阈值的优质类别索引: {good_clusters}")
