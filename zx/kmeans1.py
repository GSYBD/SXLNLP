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
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import euclidean_distances


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
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量(句子数量的平方根)
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    # 聚类中心
    cluster_centers = kmeans.cluster_centers_
    # 聚类标签
    labels = kmeans.labels_

    # 计算类内平均距离 欧式距离
    intra_cluster_dissimilarities = []
    for i in range(kmeans.n_clusters):
        cluster_vectors = vectors[labels == i]
        distances_to_center = euclidean_distances(cluster_vectors, [cluster_centers[i]])
        intra_cluster_dissimilarities.append(np.mean(distances_to_center))

    # # 计算类内平均距离 余弦相似度
    # intra_cluster_dissimilarities = []
    # for i in range(kmeans.n_clusters):
    #     cluster_vectors = vectors[labels == i]  # 转换为numpy数组以进行余弦相似度计算
    #     # 计算每个样本与聚类中心的余弦相似度
    #     similarities = cosine_similarity(cluster_vectors, [cluster_centers[i]])
    #     # 转换为不相似度（1 - 相似度）
    #     dissimilarities = 1 - similarities
    #     # 计算平均不相似度
    #     intra_cluster_dissimilarities.append(np.mean(dissimilarities))

        # 排序类内平均距离
    #print(intra_cluster_dissimilarities)
    sorted_indices = np.argsort(intra_cluster_dissimilarities) #np.argsort() 函数来获取这些不相似度值的排序索引
    print(sorted_indices)

    # 保留前N个聚类
    N = 10  # 保留一半的聚类
    # 创建一个布尔索引数组，用于过滤出我们感兴趣的标签
    keep_indices = np.isin(labels, sorted_indices[:N])

    # 使用这个布尔索引数组来过滤标签
    filtered_labels = labels[keep_indices]
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, filtered_labels):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

