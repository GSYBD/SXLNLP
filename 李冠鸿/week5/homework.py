


import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


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

def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def main():
    model = load_word2vec_model("model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    density_dict = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]
        center = kmeans.cluster_centers_[label]
        distance = cosine_distance(vector, center)
        density_dict[label].append(distance)
    for label, distance_lsit in density_dict.items():
        density_dict[label] = np.mean(distance_lsit)
    density_order = sorted(density_dict.items(), key=lambda x:x[1], reverse=True)

    for label, distance_avg in density_order:
        print("cluster %s, avg distance %f: " % (label, distance_avg))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("--------------")

def cosine_distance(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))
    vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))
    return np.sum(vec1 * vec2)

def eculid_distance(vec1, vec2):
    return np.sqrt((np.sum(np.square(vec1 - vec2))))

if __name__ == '__main__':
    main()
