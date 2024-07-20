#!/usr/bin/env python3  
#coding: utf-8


##实现基于kmeans的类内距离计算，筛选优质类别
# 聚类结束后计算类内平均距离
# 排序后，舍弃类内平均距离较长的类别


import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量（去重后）：", len(sentences))
    return sentences


def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        word_count = 0
        for word in words:
            try:
                vector += model.wv[word]
                word_count += 1
            except KeyError:
                # 忽略不在模型中的词
                continue
        if word_count > 0:
            vectors.append(vector / word_count)
    return np.array(vectors)


def calculate_cluster_inertia(vectors, labels, n_clusters):
    cluster_inertia = {}
    for i in range(n_clusters):
        cluster_points = vectors[labels == i]
        center = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        average_distance = np.mean(distances)
        cluster_inertia[i] = average_distance
    return cluster_inertia


def filter_clusters(cluster_inertia, threshold=None, num_clusters_to_keep=None):
    # 你可以根据类内平均距离设置阈值，或者保留前N个聚类
    if threshold is not None:
        filtered_clusters = {k: v for k, v in cluster_inertia.items() if v <= threshold}
    elif num_clusters_to_keep is not None:
        sorted_clusters = sorted(cluster_inertia.items(), key=lambda x: x[1])
        filtered_clusters = dict(sorted_clusters[:num_clusters_to_keep])
    else:
        raise ValueError("Either 'threshold' or 'num_clusters_to_keep' must be provided.")
    return filtered_clusters

def main():
    model_path = r"C:\Users\lixiaoxiao822\Desktop\八斗AI\第五周 词向量\week5 词向量及文本向量\model.w2v"
    sentences = load_sentence("titles.txt")
    model = load_word2vec_model(model_path)
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, random_state=42)  # 添加随机状态以确保结果可复现
    kmeans.fit(vectors)

    # 计算类内距离并排序
    cluster_inertia = calculate_cluster_inertia(vectors, kmeans.labels_, n_clusters)
    sorted_clusters = sorted(cluster_inertia.items(), key=lambda x: x[1])

    # 假设我们只保留类内平均距离最小的前5个聚类
    num_clusters_to_keep = 5
    filtered_clusters = filter_clusters(cluster_inertia, num_clusters_to_keep=num_clusters_to_keep)

    # 打印筛选后的聚类结果
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        if label in filtered_clusters:
            sentence_label_dict[label].append(sentence)

    for label, sentences in sentence_label_dict.items():
        print(f"Cluster {label} (retained):")
        for i, sentence in enumerate(sentences[:10]):
            print(f"  {i + 1}. {sentence.replace(' ', '')}")
        print("---------")

        # 如果需要，也可以打印被丢弃的聚类信息
    print("Dropped clusters:")
    for label, _ in sorted_clusters[num_clusters_to_keep:]:
        print(f"Cluster {label} with average distance {cluster_inertia[label]}")

if __name__ == "__main__":
    main()