#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
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
            sentence = line.strip()  # 删除开头结尾的空格或者换行符
            sentences.add(" ".join(jieba.lcut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r".\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)   # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)          # 进行聚类计算
    cluster_centers = kmeans.cluster_centers_

    label_sentence_distance_dict = defaultdict(list)
    for label, sentence, vector in zip(kmeans.labels_, sentences, vectors):  # 取出标签、句子与向量
        center = cluster_centers[label]
        # 计算每一个句子向量与质心的距离
        distance = np.abs(center-vector)
        label_sentence_distance_dict[label].append([sentence, distance])

    # 计算每一个cluster的类内平均距离，并与标签对应
    label_distanceAvg = []
    for label, sentences_distances in label_sentence_distance_dict.items():
        distance_sum = 0
        for sentence_distance in sentences_distances:
            distance_sum += np.mean(sentence_distance[1])
        distance_avg = distance_sum / len(sentences_distances)
        label_distanceAvg.append([label, distance_avg])

    # 根据类内平均距离排序，距离越小 聚类效果越好
    label_distanceAvg = sorted(label_distanceAvg, key=lambda x:x[1], reverse=False)

    # 展示聚类效果top10
    for i in range(10):
        label = label_distanceAvg[i][0]
        distanceAvg = label_distanceAvg[i][1]
        sentence_distance = label_sentence_distance_dict[label]
        print("cluster %s , distanceAvg %3.4f:" % (label, distanceAvg))
        for i in range(min(10, len(sentence_distance))):  #
            print(sentence_distance[i][0].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()

