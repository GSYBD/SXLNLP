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
def load_word2vec_model(path): #导入文本
    model = Word2Vec.load(path)
    return model

def load_sentence(path):#导入句子
    sentences = set()#set() 函数创建集合--无重复性
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()#strip() 方法在没有参数的情况下，会去除字符串开头和结尾的所有空白字符。这包括空格、制表符（\t）、换行符（\n）、回车符（\r）、换页符（\f）和垂直制表符（\v）
            sentences.add(" ".join(jieba.cut(sentence)))#将词链接
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []#设置空列表
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)  #100维0矩阵
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
    print(sentences)
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    print(vectors )

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for vector, label in zip(vectors, kmeans.labels_):  # 取出向量和标签
        vector_label_dict[label].append(vectors)  # 同标签的放到一起
    centers = kmeans.cluster_centers_
    label_distance = {}
    for label, vector in vector_label_dict.items():
        centers = kmeans.cluster_centers_[label]
        label_distance[label] = np.mean([np.linalg.norm(vec - centers) for vec in vectors])
    sorted_label_distance = sorted(label_distance.items(), key=lambda x: x[1], reverse=True)
    k = 10
    top_k_label = [item[0] for item in sorted_label_distance[:k]]
    for label in top_k_label:
        print(f"cluster {label} :")
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")
    print(centers)
    print(sentence_label_dict)
    print(vector_label_dict)
if __name__ == "__main__":
    main()

