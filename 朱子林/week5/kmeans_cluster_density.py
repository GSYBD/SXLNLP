#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于训练好的词向量模型进行聚类
聚类采用KMeans算法
"""

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


def load_word2vec_model(path):
    """
    加载训练好的词向量模型

    :param path: 模型文件路径
    :return: Word2Vec模型
    """
    model = Word2Vec.load(path)
    return model


def load_sentences(path):
    """
    加载并分词所有句子

    :param path: 句子文件路径
    :return: 分词后的句子集合
    """
    sentences = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences



def sentences_to_vectors(sentences, model):
    """
    将句子转换为向量

    :param sentences: 分词后的句子集合
    :param model: Word2Vec模型
    :return: 句子向量数组
    """
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # 分词后的句子，空格分开
        vector = np.zeros(model.vector_size)
        for word in words:
            if word in model.wv:
                vector += model.wv[word]
        if len(words) > 0:
            vector /= len(words)
        vectors.append(vector)
    return np.array(vectors)


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    :param vec1: 向量1
    :param vec2: 向量2
    :return: 余弦相似度
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def main():
    model = load_word2vec_model("model.w2v")
    sentences = load_sentences("titles.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)

    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    density_dict = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]
        center = kmeans.cluster_centers_[label]
        distance = cosine_similarity(vector, center)
        density_dict[label].append(distance)

    for label in density_dict:
        density_dict[label] = np.mean(density_dict[label])

    density_order = sorted(density_dict.items(), key=lambda x: x[1], reverse=True)

    for label, distance_avg in density_order:
        print(f"cluster {label} , avg similarity {distance_avg}: ")
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
