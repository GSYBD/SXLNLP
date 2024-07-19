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
from collections import OrderedDict

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

#将文本向量化 ==句子向量化
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
    model = load_word2vec_model("./model.w2v") #加载词向量模型
    sentences = load_sentence("../data/titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    sentence_vectors_dict = defaultdict(list)
    for sentence, label in zip(vectors, kmeans.labels_):  #取出句子向量和标签
        sentence_vectors_dict[label].append(sentence)         #同标签的放到一起

    # 按类内平均距离排序，距离中心越近越好
    # print(kmeans.cluster_centers_)
    sentence_avg_dict = defaultdict(list)
    for label, label_vector in sentence_vectors_dict.items():
        # print("cluster %s :" % label)
        sum_vector = 0

        for sentence_vector in label_vector:
            # print(sentence_vector)
            # 距离中心点的余弦距离
            sum_vector += np.dot(sentence_vector, kmeans.cluster_centers_[label]) \
                          / (np.linalg.norm(sentence_vector) * np.linalg.norm(kmeans.cluster_centers_[label]))
        avg_vector = sum_vector / len(label_vector)
        sentence_avg_dict[label].append(avg_vector)

    for label,val in sentence_avg_dict.items():
        sentence_avg_dict[label] = val[0]

    # 排序
    print(sentence_avg_dict)
    sentence_avg_dict = OrderedDict(sorted(sentence_avg_dict.items(), key=lambda t: t[1], reverse=True))
    print(sentence_avg_dict)

    for label, cos_num in sentence_avg_dict.items():
        print("cluster %s %s:" % (label,cos_num))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

