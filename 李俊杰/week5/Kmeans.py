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
import itertools


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
    model = load_word2vec_model(r"C:\Users\27551\Desktop\LearnAI\pythonProject\week5\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    cluster_centers_dict = {i: center for i, center in enumerate(kmeans.cluster_centers_)}  # 类别：类的质心向量
    # print(cluster_centers_dict)
    labels_dict = {i: label for i, label in enumerate(kmeans.labels_)}  # 文本下标：文本类别
    # print(labels_dict)
    dis_len = defaultdict(list)  # 类别：[文本1、文本2…………]
    for i, label in labels_dict.items():
        dis_len[label].append(i)

    ave_len = {}
    for label, lis in dis_len.items():
        sum_difference = 0
        for item in lis:
            difference = cluster_centers_dict[label] - item
            sum_difference += np.sum(difference)
        ave_len[label] = abs(sum_difference / len(lis))
    # print(ave_len)
    sorted_d = {k: v for k, v in sorted(ave_len.items(), key=lambda item: item[1])}

    for key, _ in itertools.islice(sorted_d.items(), 10):
        print("cluster %s :" % key)
        for i in range(min(10, len(sentence_label_dict[key]))):  # 随便打印几个，太多了看不过来
            print(sentence_label_dict[key][i].replace(" ", ""))
        print("---------")

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")


if __name__ == "__main__":
    main()
