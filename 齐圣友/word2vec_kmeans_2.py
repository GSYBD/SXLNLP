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
from collections import defaultdict

from sklearn.metrics.pairwise import euclidean_distances

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


def main():
    model = load_word2vec_model(r"D:\Tools\JetBrains\WorkSpace\word_embedding\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    # 初始化一个列表来存储每个聚类的类内距离
    intra_cluster_distance = []
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    for i in range(n_clusters):
        # 找到所有属于第i个聚类的点
        cluster_points = vectors[labels == i]
        # 计算这些点到聚类中心的距离
        distance = euclidean_distances(cluster_points, centers[i].reshape(1, -1))
        # 计算平均距离
        avg_distance = np.mean(distance)
        intra_cluster_distance.append(avg_distance)
    print(f"Average intra-cluster distance: {np.mean(intra_cluster_distance)}")

    # 打印每个聚类的类内距离
    for i, dist in enumerate(intra_cluster_distance):
        print(f"Cluster {i} average intra-cluster distance: {dist}")

    # 根据设置的阈值来筛选出多个优质类别
    threshold = np.mean(intra_cluster_distance)  # 设置阈值
    print(f"Threshold: {threshold}")
    good_clusters = [i for i, dist in enumerate(intra_cluster_distance) if dist < threshold]
    print(f"Good Clusters (Below Threshold): {good_clusters}")
    # 打印聚类结果
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

