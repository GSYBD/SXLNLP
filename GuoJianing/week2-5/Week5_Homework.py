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
from sklearn.metrics.pairwise import euclidean_distances

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
    model = load_word2vec_model(r"D:\人工智能学习\model.w2v") #加载词向量模型
    sentences = load_sentence(r"D:\人工智能学习\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    centers = kmeans.cluster_centers_

    # 初始化一个字典来存储每个簇的总距离和点的数量
    cluster_distances = defaultdict(lambda: {'total_distance': 0, 'num_points': 0, 'texts': []})

    # 遍历每个点和它的簇标签
    for vector, label, text in zip(vectors, kmeans.labels_,sentences):
        # 计算点到簇中心的距离
        distance = euclidean_distances([vector], [centers[label]])[0][0]
        # 更新簇的总距离和点数
        cluster_distances[label]['total_distance'] += distance
        cluster_distances[label]['num_points'] += 1
        cluster_distances[label]['texts'].append(text)

    # 计算每个簇的平均距离
    average_distances = {label: info['total_distance'] / info['num_points'] for label, info in
                         cluster_distances.items()}

    # 排序平均距离
    sorted_clusters = sorted(average_distances.items(), key=lambda x: x[1], reverse=True)

    # 输出前十个簇及其平均距离
    top_ten_clusters = sorted_clusters[:10]
    for label, avg_distance in top_ten_clusters:
        print(f"Cluster {label}: Average Distance to Center is {avg_distance:.2f}")
        # 打印该簇的部分文本
        for text in cluster_distances[label]['texts'][:10]:  # 打印前10个文本
            print(text.replace(" ", ""))

if __name__ == "__main__":
    main()

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
from sklearn.metrics.pairwise import euclidean_distances

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
    model = load_word2vec_model(r"D:\人工智能学习\model.w2v") #加载词向量模型
    sentences = load_sentence(r"D:\人工智能学习\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    centers = kmeans.cluster_centers_

    # 初始化一个字典来存储每个簇的总距离和点的数量
    cluster_distances = defaultdict(lambda: {'total_distance': 0, 'num_points': 0, 'texts': []})

    # 遍历每个点和它的簇标签
    for vector, label, text in zip(vectors, kmeans.labels_,sentences):
        # 计算点到簇中心的距离
        distance = euclidean_distances([vector], [centers[label]])[0][0]
        # 更新簇的总距离和点数
        cluster_distances[label]['total_distance'] += distance
        cluster_distances[label]['num_points'] += 1
        cluster_distances[label]['texts'].append(text)

    # 计算每个簇的平均距离
    average_distances = {label: info['total_distance'] / info['num_points'] for label, info in
                         cluster_distances.items()}

    # 排序平均距离
    sorted_clusters = sorted(average_distances.items(), key=lambda x: x[1], reverse=True)

    # 输出前十个簇及其平均距离
    top_ten_clusters = sorted_clusters[:10]
    for label, avg_distance in top_ten_clusters:
        print(f"Cluster {label}: Average Distance to Center is {avg_distance:.2f}")
        # 打印该簇的部分文本
        for text in cluster_distances[label]['texts'][:10]:  # 打印前10个文本
            print(text.replace(" ", ""))

if __name__ == "__main__":
    main()

