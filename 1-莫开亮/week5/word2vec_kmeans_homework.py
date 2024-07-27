# coding: utf-8
"""
基于训练好的词向量模型进行聚类
聚类采用Kmeans算法
"""
import math
from collections import defaultdict
import jieba
from sklearn.cluster import KMeans
import numpy as np
from gensim.models import Word2Vec


# 加载训练好的模型
def load_word2vec_model(path):
    return Word2Vec.load(path)


def load_sentence(path):
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = jieba.lcut(line.strip())
            sentences.add(tuple(sentence))
    return sentences


# 讲文本向量化, 词向量转文本向量
def sentences_to_vectors(sentences, model):
    """"
    将文本向量化
    :param sentences: 句子列表
    :param model: 词向量模型
    :return: 句子向量列表
    """
    vectors = []
    for sentence in sentences:
        # words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in sentence:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(sentence))
    return np.array(vectors)


def main():
    model = load_word2vec_model('word2vec.model')  # 加载模型
    sentences = load_sentence('titles.txt')  # 加载文本
    # print(sentences)
    vectors = sentences_to_vectors(sentences, model)  # 文本向量化
    # print(vectors)
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("问题数量：", len(sentences))
    print("指定聚类数量：", n_clusters)
    # 指定聚类数量
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 输出每个句子对应的标签
        sentence_label_dict[label].append(sentence)

    # 计算类内距离
    centers = kmeans.cluster_centers_  # 查看聚类中心
    density_dict = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]  # 获取文本向量
        center = centers[label]  # 获取聚类中心
        distance = cosine_distance(vector, center)  # 计算向量距离
        density_dict[label].append(distance)  # 将向量距离添加到字典中

    # 对于每一类，将类内所有文本到中心的向量余弦值取平均值
    for label, distance in density_dict.items():
        density_dict[label] = np.mean(distance)

    # 按平均值排序,向量夹角余弦值越接近1，距离越小
    density_dict = sorted(density_dict.items(), key=lambda x: x[1], reverse=True)

    # 输出打印
    for label, distance_avg in density_dict:
        print("cluster %s ->  distance_avg: %f" % (label, distance_avg))
        sentence = sentence_label_dict[label]
        for i in range(min(5, len(sentence))):  # 输出前5个问题
            print(''.join(map(str, sentence[i])))
        print("---------------------------------------------")
    return


# 计算向量余弦距离,向量夹角余弦值越接近1，距离越小
def cosine_distance(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def cosine_distance2(vector1, vector2):
    vec1 = vector1 / np.sqrt(np.sum(np.square(vector1)))
    vec2 = vector2 / np.sqrt(np.sum(np.square(vector2)))
    return np.sum(vec1 * vec2)


# 欧式距离
def eculid_distance(vec1, vec2):
    return np.sqrt((np.sum(np.square(vec1 - vec2))))


if __name__ == '__main__':
    main()
