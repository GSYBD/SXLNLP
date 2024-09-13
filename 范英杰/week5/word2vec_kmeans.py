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

'''
基于kmeans算法分类，计算类内距离筛选优质类别
'''
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
# 计算余弦距离
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def cosine_distance(vec1, vec2):
    return 1 - cosine_similarity(vec1, vec2)

def main():
    model = load_word2vec_model(r"/Users/serendipity/sdk/python/document/week05-词向量/week5 词向量及文本向量/model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    sentence_label_sort = {}
    # 获取余弦距离最小前10分类
    for label,vector in zip(labels,vectors):
       center = centers[label]
       val = cosine_distance(center, vector)
       sentence_label_sort[label] = val
    sorted_items = sorted(sentence_label_sort.items(), key=lambda item: item[1], reverse=False)[:10]

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        if any(label == item[0] for item in sorted_items):
            sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")
    for label, distance in sorted_items:
        print("cluster %s :" % label)
        print("similarity %s :" % distance)
        for sentence in sentence_label_dict[label]:
            print(sentence.replace(" ", ""))

if __name__ == "__main__":
    main()

