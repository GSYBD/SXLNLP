import random
import sys
import math
import numpy as np
import json
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.metrics import pairwise_distances

# 计算类内距离
def calculate_intra_cluster_distances(kmeans, vectors):
    labels = kmeans.labels_
    distances = []
    for label in np.unique(labels):
        cluster_vectors = vectors[labels == label]
        distance = pairwise_distances(cluster_vectors).sum() / (cluster_vectors.shape[0] * (cluster_vectors.shape[0] - 1) / 2)
        distances.append(distance)
    return distances

# 筛选优质类别（假设距离越小越优质）
def select_optimal_clusters(distances):
    min_distance = min(distances)
    optimal_clusters = [i for i, d in enumerate(distances) if d == min_distance]
    return optimal_clusters

def load_word2Vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            result = line.strip()
            sentences.add(" ".join(jieba.cut(result)))
    print("句子数量:", len(sentences))
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
    model = load_word2Vec_model(r"model.w2v")  #加载词向量模型
    sentences = load_sentence(r"titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences,model )  #将文本向量化

    n_clusters = int(math.sqrt(len(sentences)))  #聚类数量
    print("指定聚类数量:", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters)  #Kmeans聚类
    kmeans.fit(vectors)   #进行聚类计算

    distances = calculate_intra_cluster_distances(kmeans, vectors)
    optimal_clusters = select_optimal_clusters(distances)

    sentences_label_dict = defaultdict(list)  #根据聚类结果，
    for sentence, label in zip(sentences, kmeans.labels_):       #取出句子和标签
        sentences_label_dict[label].append(sentence)

    print("类内距离：", distances)
    print("优质类别：", optimal_clusters)

    for label, sentences in sentences_label_dict.items():       #输出聚类结果
        print("cluster %s :" % label)
        for i in range(min(20, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == '__main__':
    main()