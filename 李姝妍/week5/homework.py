import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#实现基于kmeans的类内距离计算，筛选优质类别。

def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def calculate_cluster_instance(n_clusters,vectors):
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    cluster_instances = []
    # 遍历所有簇
    for i in range(n_clusters):
        # 获取当前簇的所有点
        cluster_points = vectors[labels == i]
        # 如果有数据点，计算点到簇中心的距离
        if len(cluster_points) > 0:
            distances = distance(cluster_points, centers[i])
            cluster_instances.append(np.array(distances).mean(axis=0))
    return np.array(cluster_instances).mean(axis=0)

def distance(p1, p2):
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)


def main():
    model = load_word2vec_model(r"/李姝妍/week5/model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model)
    # 聚类数量
    n_clusters = int(math.sqrt(len(sentences)))
    cluster_instances = calculate_cluster_instance(n_clusters, vectors)  # 调用函数计算每个簇的平均距离并返回总平均距离值。

    # 找到平均距离最小的簇并打印
    min_avg_distance, best_cluster = min(cluster_instances, key=lambda x: x[0])
    print("最优的簇索引：", best_cluster)