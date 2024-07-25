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

#计算两个向量之间的欧氏距离
def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist

def main():
    model = load_word2vec_model(r"D:\八斗精品班\第五周 词向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("D:\\八斗精品班\\第五周 词向量\\week5 词向量及文本向量\\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    # print(vectors[1])

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    vector_label_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):  #取出句子和标签
        vector_label_dict[label].append(vector)
    #print(vector_label_dict)
    
    direction_to_center_avg_dict = {}
    #分别计算每个label里的每个点到对应质心的平均距离
    for labels, directions in vector_label_dict.items():
        # print(labels)
        direction_to_center_sum = 0
        for direction in directions:
            direction_to_center_sum += embedding_distance(direction, kmeans.cluster_centers_[labels])
        direction_to_center_avg = direction_to_center_sum / len(directions)
        direction_to_center_avg_dict[labels] = direction_to_center_avg
    # print(direction_to_center_avg_dict)

    dict1 = sorted(direction_to_center_avg_dict.items(),key = lambda x:x[1]) #升序
    # print(dict1)
    sort_label = [txt[0] for txt in dict1]
    print(sort_label)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    for label, sentences in sentence_label_dict.items():
        if label in sort_label[:10]:
            print("cluster %s :" % label)
            for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
                print(sentences[i].replace(" ", ""))
            print("---------")
            

if __name__ == "__main__":
    main()

