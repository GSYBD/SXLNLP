# coding: utf-8

# 实现基于kmeans的类内距离计算，筛选优质类别

from gensim.models import Word2Vec
import jieba
import numpy as np
import math
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.spatial import distance


def load_model(path):
    return Word2Vec.load(path)


def load_sentences(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量: ", len(sentence))
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
    model = load_model("model.w2v")
    sentences = load_sentences("titles.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
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
    for label, distance_list in density_dict.items():
        density_dict[label] = np.mean(distance_list)
    density_order = sorted(density_dict.items(), key=lambda x:x[1], reverse=True)

    for label, distance_avg in density_order:
        print(f"cluster {label}, avg dist {distance_avg}")
        sentences = sentence_label_dict[label]
        for i in range(min(5, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("-------------")


# 余弦距离
def cosine_distance(vector, center):
    # vec1 = vec1 / np.sqrt(np.sum(np.square(vec1)))  #A/|A|
    # vec2 = vec2 / np.sqrt(np.sum(np.square(vec2)))  #B/|B|
    # return 1 - np.sum(vec1 * vec2)

    cosine_dist = distance.cosine(vector, center)
    # eucl_dist = distance.euclidean(vector, center)  # 欧氏距离
    return cosine_dist


if __name__ == '__main__':
    main()
