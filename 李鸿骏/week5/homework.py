#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


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
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"E:\ai课程\八斗精品班\week5+词向量及文本向量\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence(r"E:\ai课程\八斗精品班\week5+词向量及文本向量\week5 词向量及文本向量\titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, n_init='auto')  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    vectors_label_dict = defaultdict(list)
    for sentence, vector, label in zip(sentences, vectors, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起
        vectors_label_dict[label].append(vector)
    # 计算每个标签的类内平均距离
    centers = kmeans.cluster_centers_  # 获取所有类别的中心向量
    avg_distance_label_dict = {}
    for i in range(n_clusters):
        center_vector = centers[i]  # 获取第i个标签的中心向量
        vectors_label = vectors_label_dict[i]  # 获取第i个标签的所有向量
        avg_distance = np.mean([np.linalg.norm(center_vector - v) for v in vectors_label])  # 计算第i个标签的类内平均向量距离
        print("cluster %s 类内平均距离：" % i, avg_distance)
        avg_distance_label_dict[i] = avg_distance
    # 对标签的类内平均距离进行排序
    avg_distance_label_dict = sorted(avg_distance_label_dict.items(), key=lambda x: x[1], reverse=True)
    # for label, avg_distance in avg_distance_label_dict:
    #     print("排序后cluster %s 类内平均距离：%s" % (label, avg_distance))
    # 舍弃类内平均距离较长的10个类别
    for label, _ in avg_distance_label_dict[:10]:
        pop_list = sentence_label_dict.pop(label)
        print("舍弃类别 %s :" % label)
        for i in range(min(10, len(pop_list))):
            print(pop_list[i].replace(" ", ""))
        print("---------")
    print("优化后类别数量为 %s" % len(sentence_label_dict))
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
