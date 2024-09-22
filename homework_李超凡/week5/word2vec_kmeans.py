import math
import jieba
import numpy as np
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


# 加载词向量模型
def load_word2vec_model(word2vec_model_path):
    model = Word2Vec.load(word2vec_model_path)
    return model


# 加载数据
def load_sentences(sentence_path):
    sentences = []
    with open(sentence_path, encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            sentences.append(" ".join(jieba.cut(sentence)))
    print("样本数量：", len(sentences))
    return sentences


# 语句转向量
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        vector_num = 0
        for word in words:
            try:
                vector += model.wv[word]
                vector_num += 1
            except KeyError:
                continue
        vector = vector / vector_num
        vectors.append(vector)
    return np.array(vectors)


# kmeans聚类
def kmeans_cluster(vectors):
    n_clusters = int(math.sqrt(len(vectors)))
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    labels_ed_dict = defaultdict(list)
    labels_index_dict = defaultdict(list)
    for i in range(len(labels)):
        labels_ed_dict[labels[i]].append(np.sqrt(np.sum((vectors[i] - centers[labels[i]]) ** 2)))
        labels_index_dict[labels[i]].append(i)
    return labels_ed_dict,labels_index_dict


# 根据类内点到中心的平均距离筛选
def cluster_select(sentence_index_ed_dict, top_key=10):
    labels = []
    avg_eds = []
    for key in sentence_index_ed_dict.keys():
        labels.append(key)
        avg_eds.append(np.mean(sentence_index_ed_dict[key]))
    select_labels = [labels[i] for i in np.argsort(np.array(avg_eds))[:top_key]]
    return select_labels


if __name__ == "__main__":
    sentences_path = "titles.txt"
    word2vec_model_path = "model.w2v"

    model = load_word2vec_model(word2vec_model_path)
    sentences = load_sentences(sentences_path)
    vectors = sentences_to_vectors(sentences, model)
    labels_ed_dict,labels_index_dict = kmeans_cluster(vectors)
    select_labels = cluster_select(labels_ed_dict)

    for  label in select_labels:
        print("*"*30)
        for idx in labels_index_dict[label]:
            print(sentences[idx])