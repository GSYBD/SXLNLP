#!/usr/bin/env python3  
#coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用KMeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)  # 使用gensim加载Word2Vec模型
    return model

# 加载并处理文本数据
def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))  # 使用jieba进行分词，将结果加入集合
    print("获取句子数量：", len(sentences))
    return sentences

# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)  # 初始化句子向量为全零
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]  # 将词向量相加
            except KeyError:
                vector += np.zeros(model.vector_size)  # 如果词不在词汇表中，用全零向量代替
        vectors.append(vector / len(words))  # 计算平均向量
    return np.array(vectors)

def main():
    # 加载词向量模型
    model = load_word2vec_model(r"model.w2v") 
    # 加载所有标题
    sentences = load_sentence("titles.txt")  
    # 将所有标题向量化
    vectors = sentences_to_vectors(sentences, model)   

    # 指定聚类数量，使用句子数量的平方根
    n_clusters = int(math.sqrt(len(sentences)))  
    print("指定聚类数量：", n_clusters)
    # 定义一个KMeans计算类
    kmeans = KMeans(n_clusters)  
    # 进行聚类计算
    kmeans.fit(vectors)          

    # 计算每个样本到其聚类中心的距离
    distances = kmeans.transform(vectors)
    labels = kmeans.labels_
    
    # 计算每个类别的平均距离
    cluster_distances = defaultdict(list)
    for label, distance in zip(labels, distances):
        cluster_distances[label].append(distance[label])
    
    # 计算每个类别的平均距离
    avg_distances = {label: np.mean(dist) for label, dist in cluster_distances.items()}
    
    # 对类别按平均距离进行排序，并舍弃类内平均距离较长的类别
    sorted_avg_distances = sorted(avg_distances.items(), key=lambda item: item[1])
    # 舍弃20%平均距离最长的类别
    cutoff_index = int(0.8 * len(sorted_avg_distances))  
    selected_labels = set(label for label, _ in sorted_avg_distances[:cutoff_index])
    discarded_labels = set(label for label, _ in sorted_avg_distances[cutoff_index:])
    
    sentence_label_dict = defaultdict(list)
    discarded_sentence_label_dict = defaultdict(list)
    
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        if label in selected_labels:
            sentence_label_dict[label].append(sentence)     # 同标签的放到一起
        else:
            discarded_sentence_label_dict[label].append(sentence)  # 被舍弃的标签放在另一组
    
    # 打印每个类别中的句子
    print("保留的类别：")
    for label, sentences in sentence_label_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")
    
    # 打印被舍弃的类别和其中的句子
    print("舍弃的类别：")
    for label, sentences in discarded_sentence_label_dict.items():
        print("discarded cluster %s :" % label)
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
