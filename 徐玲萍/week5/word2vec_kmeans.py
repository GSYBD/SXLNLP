# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

"""
kmeans 计算类别
用平均欧式距离计算哪些类别优质，距离越小越相似
"""

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


# 计算每个句子跟中心的距离的平均
def kemens_distance_sum(vector_label_dict, kmeans_cluster_centers):
    label_distance_dict = defaultdict(float)

    for label, center in enumerate(kmeans_cluster_centers):
        label_distance_dict[label] = distance_sentence_ave(center, vector_label_dict[label])

    return label_distance_dict


def distance_sentence_ave(center, sentence_vectors):
    tmp = 0
    for i in range(len(sentence_vectors)):
        tmp += distance(center, sentence_vectors[i])

    return tmp / len(sentence_vectors)


def distance(center, vector):
    tmp = 0
    for i in range(len(vector)):
        tmp += pow(vector[i] - center[i], 2)
    return pow(tmp, 0.5)


def main():
    # os.environ['OMP_NUM_THREADS'] = '8'

    model = load_word2vec_model("model.w2v")  # 加载词向量模型

    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, n_init=10)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算 ，返回kmeans的对象

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 每个类别的中心
    kmeans_cluster_centers = kmeans.cluster_centers_

    # 每个类别的向量
    vector_label_dict = defaultdict(list)
    for vector, label in zip(vectors, kmeans.labels_):
        vector_label_dict[label].append(vector)

        # 获取每个类别的欧式距离
    label_distance_dict = kemens_distance_sum(vector_label_dict, kmeans_cluster_centers)
    label_distance_dict = dict(sorted(label_distance_dict.items(), key=lambda item: item[1]))

    top = 10
    for lable, cate_distance in label_distance_dict.items():
        print("cluster %s :" % label)
        print("distance %s :" % cate_distance)

        for i in range(min(10, len(sentence_label_dict[lable]))):
            print(sentence_label_dict[lable][i].replace(" ", ""))
        print("---------")
        top -= 1
        if top < 0:
            break


if __name__ == "__main__":
    main()
