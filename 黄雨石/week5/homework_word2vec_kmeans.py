#!/usr/bin/env python3  
# coding: utf-8
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 实现基于kmeans的类内距离计算，筛选优质类别
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
                aa =model.wv[word]
                vector += model.wv[word]

            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def cosine_similarity(vector_A, vector_B):
    # 计算点积
    dot_product = np.dot(vector_A, vector_B)

    # 计算向量的范数
    norm_A = np.linalg.norm(vector_A)
    norm_B = np.linalg.norm(vector_B)

    # 计算余弦相似度
    cosine_sim = dot_product / (norm_A * norm_B)

    return cosine_sim


def main():
    model = load_word2vec_model("model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算 挑更好类，类内相似更好，公共含义越好，对每一个类，在计算类内的平均距离，舍弃距离较长的计算一个平均，越低越好
    # 计算余弦值，只看比较好的类

    sentence_label_dict = defaultdict(list)
    # 已经聚完所有的类了都有一个中心，也知道，任意两个向量距离都可以计算，
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 计算类内平均欧式距离
    # 相同类别向量放一起
    vector_acc = defaultdict(list)
    vector_avg = defaultdict(float)
    for label in kmeans.labels_:
        vector_acc[label].append(
            cosine_similarity(np.array(vectors[label]), np.array(kmeans.cluster_centers_[label])))
    #计算平均距离
    for k, v in vector_acc.items():
        vector_avg[k]=np.sum(v)/len(v)
    sorted_vector_avg = sorted(vector_avg.items(),key=lambda x:x[1])

    print("第一个向量",vectors[0],sentences)
    sorted_vector_avg = sorted_vector_avg[:10]
    print(sorted_vector_avg)
    for label, avg in sorted_vector_avg:
        for i in range(min(10, len(sentence_label_dict[label]))):  # 随便打印几个，太多了看不过来
            print(label,sentence_label_dict[label][i].replace(" ", ""))
        print("---------")



if __name__ == "__main__":
    main()

    # 示例向量
    # vector_A = np.array([1, 2, 3])
    # vector_B = np.array([4, 5, 6])
    #
    # # 计算余弦相似度
    # cos_sim = cosine_similarity(vector_A, vector_B)
    # print(f"向量 A 和 向量 B 的余弦相似度: {cos_sim}")