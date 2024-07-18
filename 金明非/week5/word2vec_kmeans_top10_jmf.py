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
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = list(load_sentence("titles.txt"))  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    sentence_distance_dict = defaultdict(dict)
    for i, label in zip(range(len(labels)), labels):
        vec1 = vectors[i]
        vec2 = centers[label]
        # distance = np.linalg.norm(vec1 - vec2)  # 计算中心到点的欧式距离， 越小越相似
        distance = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))  # 计算中心到点的余弦距离， 越大越相似
        sentence_distance_dict[label].update({sentences[i]: distance})


    distance_avg_dict = dict()
    for label, sentence_distances in sentence_distance_dict.items():
        distance_avg_dict[label] = np.mean(list(sentence_distances.values()))  # 计算各类的平均距离

    # # sorted_distance_avg = sorted(distance_avg_dict.items(), key=lambda x: x[1])[0:10]  # 各类平均欧式距离排序并取前10个
    sorted_distance_avg = sorted(distance_avg_dict.items(), key=lambda x: x[1], reverse=True)[0:10]  # 各类平均余弦距离排序并取前10个

    for label, _ in sorted_distance_avg:  # 输出聚类效果好的前10个族
        print("cluster %s :" % label)
        # sorted_sentences = sorted(sentence_distance_dict[label].items(), key=lambda x: x[1], reverse=True)[0:10] # 对类内点到中心距离排序并取前十输出
        # for sentence,_ in sorted_sentences:
        #     print(sentence.replace(" ", ""))
        sentences = list(sentence_distance_dict[label].keys())
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
