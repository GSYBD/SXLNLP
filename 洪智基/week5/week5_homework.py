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

# 计算两个向量之间的余弦距离（1 - 余弦相似度）
def cos_distance(vec1, vec2):
    return 1 - cosine_similarity([vec1], [vec2])[0][0]


# 计算类内平均距离
def calculate_intracluster_distance(vectors, labels, distance_func=cos_distance):
    cluster_distances = {}
    for label in set(labels):
        cluster_vectors = [vectors[i] for i, lbl in enumerate(labels) if lbl == label]
        if len(cluster_vectors) > 1:  # 至少需要两个向量来计算距离
            distances = [distance_func(v1, v2) for i, v1 in enumerate(cluster_vectors) for v2 in
                         cluster_vectors[i + 1:]]
            cluster_distances[label] = np.mean(distances)
        else:
            cluster_distances[label] = 0  # 单个向量的类内距离为0
    return cluster_distances

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
    try:
        model = load_word2vec_model(r"D:\HZJ\课件\week5 词向量及文本向量\model.w2v")#加载词向量模型，已重新训练为200维
        sentences = load_sentence("titles.txt") #加载所有标题
    except FileNotFoundError:
        print("文件未找到，请检查路径是否正确")
        exit(1)
    except Exception as e:
        print("加载文件或模型时发生错误：", e)
        exit(1)

    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    # print(kmeans.cluster_centers_)

    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

    # 计算类内平均距离
    intracluster_distances = calculate_intracluster_distance(vectors, kmeans.labels_)

    # 根据类内平均距离排序并打印
    sorted_clusters = sorted(intracluster_distances.items(), key=lambda x: x[1], reverse=True)

    # 设定一个阈值或保留一定数量的聚类
    # 这里简单地保留类内平均距离低于某个阈值的聚类，或者保留前N个聚类
    threshold = 0.3  # 设定一个合理的阈值
    filtered_clusters = {label: sentence_label_dict[label] for label, dist in sorted_clusters if dist <= threshold}

    # 打印过滤后的聚类结果
    for label, sentences in filtered_clusters.items():
        print("Filtered cluster %s (avg. distance: %.3f):" % (label, intracluster_distances[label]))
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

