import numpy as np
from sklearn.cluster import KMeans

def Kmeans_cluster_inner_distance(n_clusters, vectors):
    Kmeans = KMeans(n_clusters).fit(vectors)
    labels = Kmeans.labels_
    centers = Kmeans.cluster_centers_
    print(centers)
    cluster_inner_distance = []
    for i in range(centers.shape[0]):
        cluster_point = vectors[labels == i]
        distance = np.linalg.norm(cluster_point - centers[i], axis=1)
        avg_distance = np.mean(distance)
        cluster_inner_distance.append(avg_distance)
    return cluster_inner_distance

vectors = np.random.rand(10000, 100)
n_clusters = 40
inner_distance = Kmeans_cluster_inner_distance(n_clusters, vectors)
dict_key =list(range(1, n_clusters + 1))
cluster_dict = dict(zip(dict_key, inner_distance))
good_cluster = sorted(cluster_dict.items(), key=lambda item: item[1])[:20]
print("优质簇类,类内平均距离：", good_cluster)
good_cluster_label = [item[0] for item in good_cluster]
print("优质簇类：", good_cluster_label)
