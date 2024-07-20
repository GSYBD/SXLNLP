#!/usr/bin/env python3
# coding: utf-8

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
def compute_intra_cluster_distance(vectors, labels, centroids):
    cluster_distances = defaultdict(list)
    for vector, label in zip(vectors, labels):
        distance = np.linalg.norm(vector - centroids[label])
        cluster_distances[label].append(distance)

    average_distances = {label: np.mean(distances) for label, distances in cluster_distances.items()}
    return average_distances


def main():
    model = load_word2vec_model(r"E:\BaiduDownload\八斗精品班\第五周词向量\week5词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence(r"E:\BaiduDownload\八斗精品班\第五周词向量\week5词向量及文本向量\titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    centroids = kmeans.cluster_centers_
    average_distances = compute_intra_cluster_distance(vectors, kmeans.labels_, centroids)

    sorted_clusters = sorted(average_distances.items(), key=lambda item: item[1])
    print("按类内距离排序的聚类：", sorted_clusters)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    for label, avg_distance in sorted_clusters:
        print(f"Cluster {label} (average intra-cluster distance: {avg_distance:.2f}):")
        for i in range(min(10, len(sentence_label_dict[label]))):  # 随便打印几个，太多了看不过来
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
