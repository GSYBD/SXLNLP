#!/usr/bin/env python3  
#coding: utf-8
#作业：筛选出距离较短的类 可以用欧式距离、余弦距离或其他距离，距离短的几个类

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import heapq
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型

#向量的余弦距离
def CosineDistance(x, y):
    x=np.array(x)
    y=np.array(y)
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

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
    model = load_word2vec_model(r"D:\xintianli\4.zuoye\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    cluster_distance = []
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 计算向量的余弦距离
    for i in range(n_clusters):
        cluster_points = vectors[labels == i]  #找到所有属于第i个聚类的点
        distance = CosineDistance(cluster_points, centers[i])
        avg_distance = np.mean(distance)  #计算平均距离
        cluster_distance.append(avg_distance)

    # for i, dist in enumerate(cluster_distance):
    #     print(f"Cluster {i} average cluster distance: {dist}")

    # 找到最接近的5个类 余弦距离越大向量越接近
    largest = min(heapq.nlargest(6, cluster_distance))
    good_clusters = [i for i, dist in enumerate(cluster_distance) if dist > largest]


    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        if label in good_clusters: #筛选出比较好的类
            print("cluster %s :" % label)
            print(f"类内平均余弦间距: {cluster_distance[label]}")
            print(f"类内包含句子数量: {len(sentences)}","\n")
            for i in range(min(5, len(sentences))):  #随便打印几个，太多了看不过来
                print(sentences[i].replace(" ", ""))
            print("---------")

if __name__ == "__main__":
    main()

