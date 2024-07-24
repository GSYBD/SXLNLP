import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def k_means(data, k, max_iterations=100):
    #选择k个初始质心
    centroids = data[np.random.choice(data.shape[0], size=k, replace=False)]
    
    for _ in range(max_iterations):
        # 分配每个数据点到最近的质心
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        
        # 更新质心位置为分配给它的所有点的平均值
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # 如果质心不再改变，则停止迭代
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
centroids, labels = k_means(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.5)
plt.show()