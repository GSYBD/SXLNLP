#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
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
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

# 计算类内距离
def calculate_intra_cluster_distances(vectors, labels, distance_method='euclidean'):
    if distance_method == 'euclidean':
        distance_func = euclidean_distances
    elif distance_method == 'cosine':
        distance_func = cosine_distances
    else:
        raise ValueError("Invalid distance method. Choose 'euclidean' or 'cosine'")
    cluster_distances = []
    for label in np.unique(labels):
        cluster_vectors = vectors[labels == label]
        if len(cluster_vectors) > 1:
            distances = distance_func(cluster_vectors)
            mean_distance = np.mean(distances)
            cluster_distances.append((label, mean_distance))
    return cluster_distances

def main():
    model = load_word2vec_model(r"E:\BaiduNetdiskDownload\八斗预科资料\课件\week5 词向量及文本向量\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    # 计算类内距离
    cluster_distances = calculate_intra_cluster_distances(vectors, kmeans.labels_)
    cluster_distances.sort(key=lambda x: x[1])

    # 舍弃类内平均距离较长的类别
    threshold = cluster_distances[int(len(cluster_distances) * 0.5)][1]
    selected_clusters = [cluster[0] for cluster in cluster_distances if cluster[1] <= threshold]

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        if label in selected_clusters:
            sentence_label_dict[label].append(sentence)

    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()