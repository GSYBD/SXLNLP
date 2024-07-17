#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import os
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

# 获取当前目录
def get_current_path(file = ''):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    return os.path.join(current_dir, file)

def main():
    model = load_word2vec_model(get_current_path('model.w2v')) #加载词向量模型
    sentences = load_sentence(get_current_path('titles.txt'))  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for idx,(sentence, label) in enumerate(zip(sentences, kmeans.labels_)):  #取出句子和标签
        # 计算每个点到中心点的距离，并放进dict
        dist = np.linalg.norm(vectors[idx] - kmeans.cluster_centers_[label]) 
        sentence_label_dict[label].append([sentence,dist])         #同标签的放到一起

    # 每个标签内文本做距离排序，asc
    for label, sentences in sentence_label_dict.items():
        sentences.sort(key=lambda subarray : subarray[1])
        sentence_label_dict[label] = [sentences,np.mean([sentence[1] for sentence in sentences])]

    # 对每个标签所有文本的距离求均值，再排序，asc。以便找出相似度高的分类
    sorted_items = sorted(sentence_label_dict.items(),key=lambda item: item[1][1])


    for i in range(10):
        print('当前分类：',sorted_items[i][0])
        print('当前分类，所有文本到中心点距离均值：',sorted_items[i][1][1])
        for idx in range(min(10,len(sorted_items[i][1][0]))):
            print(sorted_items[i][1][0][idx][0].replace(' ',''),'距离：',sorted_items[i][1][0][idx][1])
        print('--------------------------')
if __name__ == "__main__":
    main()

