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
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from collections import defaultdict


#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()  # 创建空的集合  { 1,2,3,4 }
    with open(path, encoding="utf8") as f:
        for line in f:  # 对每一行进行遍历
            sentence = line.strip()  # 空格切分  '新增资金入场 沪胶强势创年内新高'
            sentences.add(" ".join(jieba.cut(sentence)))  # jieba分词,对一行数据进行切分
    print("获取句子数量：", len(sentences))
    return sentences  # {'卡未 离身 钱 却 被盗   江苏 ATM 出现 冒牌 读卡器', '梁博 20 日 央视 直播 再现 live 实力   冲击 冠军',...}
# # 举例
# sentences = load_sentence("titles.txt")
# print(sentences)

#将文本向量化 词向量到文本向量的转化
def sentences_to_vectors(sentences, model):
    vectors = []   # 创建空的列表
    for sentence in sentences: # 集合 对每一个进行遍历
        words = sentence.split()  #sentence是分好词的，空格分开  将当前句子按空格分割成单词，并将结果存储在words列表中。
        vector = np.zeros(model.vector_size)  # vector_size向量  # 创建一个全零向量，大小与词向量模型的维度一致。
        #所有词的向量相加求平均，作为句子向量
        for word in words: # 对集合中的一个句子 进行遍历
            try:
                vector += model.wv[word]  #  尝试从词向量模型中获取单词的词向量，并将结果累加到向量中
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size) #  # 如果单词不在词向量模型中，则将全零向量累加到向量中。
        vectors.append(vector / len(words))  # 平均整句话的向量 # 将句子向量除以句子长度（单词数量），得到平均向量。
    return np.array(vectors) # # 将句子向量列表转换为NumPy数组，并返回结果。


def main():
    model = load_word2vec_model(r"F:\Desktop\work_space\badou\八斗课程\week5 词向量及文本向量\model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model) # #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences))) #  #指定聚类数量 42 该行代码用于确定 K-means 算法的聚类数量。它计算句子数量的平方根，并将其转换为整数。平方根值被用作启发式方法，用于估计合理的聚类数量。
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    cluster_centers = kmeans.cluster_centers_  # 取出质心
    avg_centroid = np.mean(cluster_centers, axis=0)  # 计算质心的平均值

    sentence_label_dict = defaultdict(list) # # 该行代码创建了一个 defaultdict 对象，用于按照句子的聚类标签将句子进行分组。
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    print("平均质心：", avg_centroid)
    print("小于平均质心的类别：")
    for label, sentences in sentence_label_dict.items(): # 质心遍历
        centroid = cluster_centers[label]
        if np.linalg.norm(centroid - avg_centroid) < np.linalg.norm(avg_centroid):
            print("Cluster %s :" % label)
            for i in range(min(10, len(sentences))):
                print(sentences[i].replace(" ", ""))
            print("---------")

if __name__ == "__main__":
    main()
