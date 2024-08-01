import numpy as np
import random

class KMeansClusterer:
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self._initialize_centroids(ndarray, cluster_num)

    def cluster(self):
        while True:
            clusters = self._assign_clusters(self.ndarray, self.points)
            new_points = self._calculate_new_centroids(clusters)
            if np.array_equal(self.points, new_points):
                break
            self.points = new_points
        intra_distances = self._calculate_intra_distances(clusters)
        return clusters, self.points, intra_distances

    def _initialize_centroids(self, ndarray, cluster_num):
        indices = random.sample(range(ndarray.shape[0]), cluster_num)
        return ndarray[indices]

    def _assign_clusters(self, data, centroids):
        clusters = [[] for _ in range(self.cluster_num)]
        for point in data:
            distances = [self._euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid_index = np.argmin(distances)
            clusters[closest_centroid_index].append(point)
        return clusters

    def _calculate_new_centroids(self, clusters):
        return np.array([np.mean(cluster, axis=0) for cluster in clusters])

    def _calculate_intra_distances(self, clusters):
        intra_distances = []
        for i, cluster in enumerate(clusters):
            distances = [self._euclidean_distance(point, self.points[i]) for point in cluster]
            intra_distances.append(np.sum(distances))
        return intra_distances

    def _euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def filter_quality_clusters(self, clusters, intra_distances, threshold):
        quality_clusters = [cluster for cluster, distance in zip(clusters, intra_distances) if distance < threshold]
        return quality_clusters


x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
clusters, centers, intra_distances = kmeans.cluster()


threshold = np.mean(intra_distances)
quality_clusters = kmeans.filter_quality_clusters(clusters, intra_distances, threshold)

print("Clusters:", clusters)
print("Centers:", centers)
print("Intra-cluster distances:", intra_distances)
print("Quality Clusters:", quality_clusters)
