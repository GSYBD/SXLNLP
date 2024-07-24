import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min

# 数据准备
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
})

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)

# 获取每个点的类标
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 计算类内距离
intra_distances = []
for i in range(len(centroids)):
    cluster_points = data_scaled[labels == i]
    centroid = centroids[i].reshape(1, -1)
    distances = np.sqrt(((cluster_points - centroid) ** 2).sum(axis=1))
    intra_distance = distances.sum()
    intra_distances.append(intra_distance)

# 筛选优质类别（类内距离越小，质量越高）
quality_threshold = np.mean(intra_distances)  # 设定阈值为平均类内距离
high_quality_clusters = [i for i, dist in enumerate(intra_distances) if dist < quality_threshold]

# 输出结果
print("类内距离：", intra_distances)
print("优质类别：", high_quality_clusters)