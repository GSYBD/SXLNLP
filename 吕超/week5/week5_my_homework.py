# -*- encoding: utf-8 -*-
'''
week5_my_homework.py
Created on 2024/7/17 20:29
@author: Allan Lyu
实现基于kmeans的类内距离计算，筛选优质类别
'''

# !/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法

import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


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
            sentences.add(" ".join(jieba.cut(sentence)))  # lyu: 分词操作
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #
        for word in words:
            try:
                vector += model.wv[word]
                # vector += model.wv[word] * tfidf(word)  # lyu: 同时考虑词的相似性及重要性
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    # model = load_word2vec_model(r"F:\Desktop\work_space\badou\八斗课程\week5 词向量及文本向量\model.w2v") #加载词向量模型
    model = load_word2vec_model(
        r"D:\my_study\4_八斗AI\0_八斗精品班\5_第5周_词向量及文本向量\week5 词向量及文本向量\model.w2v")  # lyu: 1.加载词向量模型, 需要提前准备
    sentences = load_sentence("titles.txt")  # 2.加载所有标题, 做jieba分词处理
    vectors = sentences_to_vectors(sentences, model)  # 3.将所有标题向量化

    # 4.进行聚类
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类, lyu: 引用sklearn.cluster
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)

    # lyu: 获取每个簇的中心点
    centers = kmeans.cluster_centers_
    # lyu: 计算每个簇的类内平均距离
    distances = []
    for i in range(n_clusters):
        distances = [np.mean(np.linalg.norm(centers[i] - vectors[j])) for j in range(len(vectors)) if
                     kmeans.labels_[j] == i]
    print("每个簇的类内平均距离：", distances)
    # lyu: 按照类内平均距离由低到高排序，取前10个簇
    sorted_indices = np.argsort(distances)[:10]
    print("前10个簇的索引：", sorted_indices)

    for i in sorted_indices:
        for sentence, label in zip(sentences, kmeans.labels_):
            if label == i:
                sentence_label_dict[label].append(sentence)
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
