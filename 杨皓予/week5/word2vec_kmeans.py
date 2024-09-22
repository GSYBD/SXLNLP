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


def main():
    model = load_word2vec_model(r"D:\资料\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, n_init='auto')  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算
    print(vectors.shape)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    density_dict = defaultdict(list)
    sentence_label_dict = defaultdict(list)
    for i, (sentence, label) in enumerate(zip(sentences, labels)):
        dist = np.linalg.norm(vectors[i] - centers[label])  # 计算当前句子与聚类中心的距离
        sentence_label_dict[label].append([sentence, dist])

    for label, sentences in sentence_label_dict.items():
        for sentence, dist in sentences:
            density_dict[label].append(dist)

    for label, dist_list in density_dict.items():
        dist_mean = np.mean(dist_list)
        density_dict[label] = dist_mean

    density_order = sorted(density_dict.items(), key=lambda x: x[1], reverse=False)
    for label, dist_mean in density_order:  # 取出句子和标签
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentence_label_dict[label][i][0].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
