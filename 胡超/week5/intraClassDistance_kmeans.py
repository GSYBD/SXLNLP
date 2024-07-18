# -*- coding: utf-8 -*-
"""
author: Chris Hu
date: 2024/7/18
desc:
sample
"""

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
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


# 计算样本点到聚类中心的欧式距离
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


# 计算样本点到聚类中心的余弦距离
def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0  # Vectors are all zeros
    return 1 - (dot_product / (norm1 * norm2))


def calc_intra_class_distance(vectors_label_dict, centers, calc_func):
    # 计算类内平均距离
    distances = dict()
    for label, vectors in vectors_label_dict.items():
        distance_sample_center = []
        for i in range(len(vectors)):
            vc1 = vectors[i]
            vc2 = centers[label]
            distance_sample_center.append(calc_func(vc1, vc2))
        distances_label_mean = np.mean(distance_sample_center)
        distances[label] = distances_label_mean
    distances = sorted(distances.items(), key=lambda x: x[1])
    print("类内平均距离：", distances)
    return distances


def main():
    model = load_word2vec_model(r"./model.w2v")  # 加载词向量模型
    sentences = load_sentence(r"./titles.txt")  # 加载所有标题,r左右防止转义
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    centers = kmeans.cluster_centers_  # 获取聚类中心
    print("聚类中心：", centers)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    # 将向量与类别放在字典里
    vectors_label_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):
        vectors_label_dict[label].append(vector)

    distances = calc_intra_class_distance(vectors_label_dict, centers, cosine_distance)

    premium_category = {distance[0]: distance[1] for distance in distances if distance[1] < 0.05}
    premium_keys = premium_category.keys()
    # inferior_category = {distance[0]: distance[1] for distance in distances if distance[1] > 0.8}
    # inferior_keys = inferior_category.keys()

    for label, sentences in sentence_label_dict.items():
        if label in premium_keys:
            desc = f"Premium cluster {label}, intra-class distance {premium_category[label]} :"
            print(desc)
            for i in range(min(10, len(sentences))):
                print(sentences[i].replace(" ", ""))
            print("-" * 44)


if __name__ == "__main__":
    main()
