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
#加载词向量
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence))) #把输入的文本做一下分词
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化，词向量到文本向量的转化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word] #这是词向量的模型，就像一个字典一样，把词向量取出，加到整句话的向量之上
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"E:\badouAI\NLP\第五周 词向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #待聚类的文本，加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题转化为向量，向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)   #相当于kmeans做计算的过程，把所有的向量传进去就可以了       #进行聚类计算

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    distance_dict = defaultdict(float)
    distance = 0

    for sentence, label in zip(sentences, kmeans.labels_):
        #kmeans有个labels_的方法，就是每一个句子对应的标签，每句话会打上是第0类，第1类，第2类，第3类
        # 取出句子和标签
        sentence_label_dict[label].append(sentence) #把第0类的放到一个列表里，把第1类的放到一个列表里，以此类推  #同标签的放到一起

    for vector, label in zip(vectors,kmeans.labels_):
        vector_label_dict[label].append(vector)

    for label, vectors in vector_label_dict.items():
        for i in range(len(vectors)):
            # 句向量
            p1 = vectors[i]
            # 质心
            p2 = kmeans.cluster_centers_[label]
            # 计算余弦距离 累加
            distance += np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
        # 计算每个类别的平均距离
        distance_dict[label] = distance / (len(vectors) + 0.01)  # +0.01防止除0错误
        # 根据距离从小到大排序,取前10个
    sorted_distance_dict = dict(sorted(distance_dict.items(),key=lambda x: x[1])[:10])
    print('Top 10 clusters which average distance are closest：')
    for label, dis in sorted_distance_dict.items():
        print("cluster %s :" % label)
        print('distance',dis)
        print("---------")

if __name__ == "__main__":
    main()
