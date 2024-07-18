#!/usr/bin/env python3  
# coding: utf-8

# 实现基于KMeans的类内距离计算，筛选优质类别
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics import euclidean_distances

# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

# 加载并分词句子
def load_sentences(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

# 将句子向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # 分词后句子用空格分开
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model_path = "model.w2v"
    sentences_path = "titles.txt"
    
    model = load_word2vec_model(model_path)  # 加载词向量模型
    sentences = load_sentences(sentences_path)  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    num_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", num_clusters)
    kmeans = KMeans(num_clusters)  # 定义KMeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 计算类内平均距离
    intra_cluster_distances = []
    for i in range(kmeans.n_clusters):
        cluster_vectors = vectors[labels == i]
        distances_to_center = euclidean_distances(cluster_vectors, [cluster_centers[i]])
        intra_cluster_distances.append(np.mean(distances_to_center))

    sorted_indices = np.argsort(intra_cluster_distances)  # 获取不相似度值的排序索引
    print("类内平均距离排序索引：", sorted_indices)

    # 保留前N个聚类
    N = 10
    keep_indices = np.isin(labels, sorted_indices[:N])
    filtered_labels = labels[keep_indices]

    sentence_cluster_dict = defaultdict(list)
    for sentence, label in zip(sentences, filtered_labels):
        sentence_cluster_dict[label].append(sentence)
    
    for label, cluster_sentences in sentence_cluster_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(cluster_sentences))):
            print(cluster_sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
