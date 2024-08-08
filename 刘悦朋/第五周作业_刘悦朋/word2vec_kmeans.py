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


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    # 计算类内平均距离
    label_distance_dict = defaultdict()  # 用于保存根据类计算得到的距离
    label_dict = defaultdict()  # 用于保存各类中的项数
    # 遍历 1796 组句向量及其标签
    for vector, label in zip(vectors, kmeans.labels_):
        # 累加 1796 组样本的欧氏距离
        label_distance_dict[label] = label_distance_dict.get(label, 0) + np.sqrt(np.sum((vector - kmeans.cluster_centers_[label]) ** 2))
        # 累加 1796 组样本各类对应的样本数
        label_dict[label] = label_dict.get(label, 0) + 1

    # 遍历 42 组标签
    for label in label_distance_dict.keys():
        # 将各组标签对应的欧式距离与样本数相除, 求均值
        label_distance_dict[label] = label_distance_dict[label] / label_dict[label]

    # 根据 42 组标签的平均欧式距离按升序排序
    sort = sorted(label_distance_dict.items(), key=lambda e:e[1])

    # 记录排名
    rank = 1

    # 遍历排名后的 42 组标签及其平均欧式距离
    for label, distance in sort:
        print("cluster {}, ranks {}, with distance as {}:".format(label, rank, distance))
        rank += 1
        for i in range(min(10, len(sentence_label_dict[label]))):
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
