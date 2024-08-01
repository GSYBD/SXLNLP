#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 计算类内距离
def compute_intra_cluster_distance(X, labels, centroids):
    intra_cluster_distances = []
    for i in range(len(centroids)):
        cluster_points = X[labels == i]
        centroid = centroids[i]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        mean_distance = np.mean(distances)
        intra_cluster_distances.append((i, mean_distance))
    return intra_cluster_distances


# 筛选出优质类别
def filter_good_clusters(intra_cluster_distances, threshold):
    good_clusters = [cluster for cluster, distance in intra_cluster_distances if distance < threshold]
    return good_clusters


# 将同标签的标题放到一起
def group_titles_by_label(titles, labels):
    label_to_titles = {}
    for title, label in zip(titles, labels):
        if label not in label_to_titles:
            label_to_titles[label] = []
        label_to_titles[label].append(title)
    return label_to_titles


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # 计算每个类别的平均距离
    intra_cluster_distances = compute_intra_cluster_distance(vectors, labels, centroids)
    # 设置类内距离的阈值
    threshold = 0.6  # 可以根据实际情况调整阈值
    good_clusters = filter_good_clusters(intra_cluster_distances, threshold)
    label_to_titles = group_titles_by_label(sentences, labels)
    for cluster in good_clusters:
        print(f"Cluster {cluster}:")
        for title in label_to_titles[cluster]:
            print(f"{title}")

if __name__ == "__main__":
    main()
