"""
基于word2vec计算kmeans
"""
import math

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

def load_w2v(model_path):
    model = Word2Vec.load(model_path)
    return model

def load_sentences(corpus_path):
    corpus = set()
    with open(corpus_path, encoding='utf8') as f:
        for line in f:
            corpus.add(" ".join(jieba.lcut(line.strip())))
    return corpus

def sentence_to_vector(sentences,model:Word2Vec):
    vectors = []
    for sentence in sentences:
        vector = np.zeros(model.vector_size)
        words = sentence.split()
        for word in words:
            try:
                vector += model.wv.get_vector(word)
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


if __name__ == '__main__':
    model = load_w2v('../model.w2v')
    sentences = load_sentences('../titles.txt')
    vectors = sentence_to_vector(sentences, model)
    n_clusters = int(math.sqrt(len(sentences)))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(vectors)

    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    label_sentence_dict = defaultdict(list)
    for label, sentence in zip(labels,sentences):
        label_sentence_dict[label].append(sentence)

    # for label,sentence in label_sentence_dict.items():
    #     print("-----",label)
    #     for s in sentence[:min(10,len(sentence))]:
    #         print(s.replace(" ",""))

    # 计算类内平均距离
    cluster_points_mean_distance = {}
    for index, center_point in enumerate(cluster_centers):
        # 获取这个中心点的所有点
        center_points = vectors[labels == index]
        # 计算这些点距离中心点的距离
        distance_center = np.linalg.norm(center_points - center_point,axis=-1)
        # 计算这些距离的平均距离
        mean_distance = np.mean(distance_center)

        cluster_points_mean_distance[index] = mean_distance

    # 排序
    cluster_points_mean_distance = sorted([(index,distance) for index,distance in cluster_points_mean_distance.items()], key=lambda x:x[1])

    for label, distance in cluster_points_mean_distance:
        print("label=",label)
        sentences = label_sentence_dict[label][:10]
        for sentence in sentences:
            print(sentence.replace(" ",""))





