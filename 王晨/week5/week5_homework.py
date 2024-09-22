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

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    avg_distances = []
    distances = []
    for label, sentences in sentence_label_dict.items():
        cluster_vectors = sentences_to_vectors(sentences, model)
        cluster_centers = kmeans.cluster_centers_
        for cluster_vec in cluster_vectors:
            for cluster_cent in cluster_centers:
                distances.append(np.linalg.norm(cluster_vec - cluster_cent))
        avg_distance = np.mean(distances)
        avg_distances.append((label, avg_distance))
        # 按类内平均距离排序，保留前20个
        avg_distances_sorted = sorted(avg_distances, key=lambda x: x[1])
        top_k = 20
        count = 0
        for label, avg_distance in avg_distances_sorted[:top_k]:
            count += 1
            print(f"第{count}个   Cluster {label}: Average distance = {avg_distance:.4f}")
            for i in range(min(10, len(sentence_label_dict[label]))):
                print(sentence_label_dict[label][i].replace(" ", ""))
            print("---------")

if __name__ == "__main__":
    main()